from MultiPlanarUNet.callbacks import init_callback_objects
from MultiPlanarUNet.logging import ScreenLogger
from MultiPlanarUNet.evaluate import loss_functions
from MultiPlanarUNet.evaluate import metrics as custom_metrics
from MultiPlanarUNet.callbacks import SavePredictionImages, Validation, \
                                      FGBatchBalancer, DividerLine, \
                                      LearningCurve
from MultiPlanarUNet.errors import raise_non_sparse_metric_or_loss_error
from MultiPlanarUNet.utils import ensure_list_or_tuple

import os
import numpy as np
from tensorflow.keras import optimizers, losses
from tensorflow.keras import metrics as TF_metrics
from multiprocessing import cpu_count
from tensorflow.python.framework.errors_impl import ResourceExhaustedError, \
                                                    InternalError


class Trainer(object):
    """
    Handles initialization and logging of model fitting sessions.
    """
    def __init__(self, model, org_model=None, logger=None):
        self.model = model
        self.logger = logger if logger is not None else ScreenLogger()

        # Extra reference to original (non GPU-distributed) model
        self.org_model = org_model or model

    def compile_model(self, optimizer, optimizer_kwargs, loss, metrics, **kwargs):
        # Initialize optimizer
        optimizer = getattr(optimizers, optimizer)
        optimizer = optimizer(**optimizer_kwargs)

        # Make sure sparse metrics and loss are specified sparse
        metrics = ensure_list_or_tuple(metrics)
        loss = ensure_list_or_tuple(loss)
        for i, m in enumerate(metrics + loss):
            if "sparse" not in m:
                raise_non_sparse_metric_or_loss_error()

        # Initialize loss(es)
        loss_list = []
        for l in loss:
            try:
                loss_list.append(getattr(losses, l))
            except AttributeError:
                import inspect
                l = getattr(loss_functions, l)
                if inspect.isclass(l):
                    loss_list.append(l(logger=self.logger, **kwargs))
                else:
                    loss_list.append(l)
        loss = loss_list

        # Find metrics both from standard keras.metrics module and own custom
        init_metrics = []
        for m in metrics:
            try:
                init_metrics.append(getattr(TF_metrics, m))
            except AttributeError:
                import inspect
                metric = getattr(custom_metrics, m)
                if inspect.isclass(metric):
                    metric = metric(logger=self.logger, **kwargs)
                init_metrics.append(metric)

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=loss,
                           metrics=init_metrics)
        self.logger("Optimizer:   %s" % optimizer)
        self.logger("Loss funcs:  %s" % loss)
        self.logger("Metrics:     %s" % init_metrics)
        return self

    def fit(self, train, val, callbacks, n_epochs, train_im_per_epoch,
            val_im_per_epoch, hparams, batch_size=8, verbose=1, init_epoch=0,
            no_im=False, use_multiprocessing=False, val_ignore_class_zero=True,
            **unused_fit_kwargs):

        # Crop labels?
        if hasattr(self.model, "label_crop"):
            train.label_crop = self.model.label_crop
            val.label_crop = self.model.label_crop

        if type(train).__name__ == "MultiTaskSequence":
            self.logger("-- Skipping saving images (not yet implemented for"
                        " MultiTaskSequences).")
            no_im = True

        # Save a few images to disk for inspection
        if no_im:
            self.logger("No images saved (--no_images flag is set)")
        else:
            from MultiPlanarUNet.utils.plotting import save_images
            im_path = os.path.join(self.logger.base_path, "images")
            save_images(train, val, im_path, self.logger)

        # Start fitting
        fitting = True
        while fitting:
            try:
                self._fit_loop(train, val, batch_size, n_epochs, verbose,
                               callbacks, init_epoch, no_im, train_im_per_epoch,
                               val_im_per_epoch, hparams, use_multiprocessing,
                               val_ignore_class_zero)
                fitting = False
            except (ResourceExhaustedError, InternalError):
                # Reduce batch size
                batch_size -= 2
                hparams["fit"]["batch_size"] = batch_size
                self.logger("\n\n[MEMORY ERROR] Reducing batch size "
                            "by 2 (now %i)" % batch_size)
                if batch_size < 1:
                    self.logger("[ERROR] Batch size negative or zero!")
                    fitting = False
            except KeyboardInterrupt:
                fitting = False
            except Exception as e:
                self.logger(e)
                raise e

        try:
            if train.image_pair_loader.queue:
                train.image_pair_loader.queue.stop()
            if val.image_pair_loader.queue:
                val.image_pair_loader.queue.stop()
        except AttributeError:
            # Multi-tasking, train.image_pair_loader will be a list
            # TODO: Make all sequences store a reference to the queue
            pass

        self.logger("Training stopped.")
        self.logger.print_calling_method = True
        return self.model

    def _fit_loop(self, train, val, batch_size, n_epochs, verbose, callbacks,
                  init_epoch, no_im, train_im_per_epoch, val_im_per_epoch,
                  hparams, use_multiprocessing, val_ignore_class_zero):

        if hasattr(train, "batch_size"):
            # Update batch size on generators (needed after OOM error->reduced
            # batch size)
            train.batch_size = batch_size

        # Get number of steps per train epoch
        if train_im_per_epoch:
            train_steps = int(np.ceil(train_im_per_epoch/batch_size))
        else:
            train_steps = len(train)

        self.logger("Using %i steps per train epoch (total batches=%i)" %
                    (train_steps, len(train)))

        if val is None or len(val) == 0:
            val = None
            callbacks = [c for c in callbacks if not any("val" in s for s in [str(v) for v in c["kwargs"].values()])]
        else:
            val.batch_size = batch_size
            if val_im_per_epoch:
                val_steps = int(np.ceil(val_im_per_epoch/batch_size))
            else:
                val_steps = len(val)
            self.logger("Using %i steps per validation epoch "
                        "(total batches=%i)" % (val_steps, len(val)))

            # Add validation callback
            # IMPORTANT: Should be first in callbacks list as other CBs may
            # depend on the validation metrics/loss
            validation = Validation(val, val_steps, logger=self.logger,
                                    verbose=verbose,
                                    ignore_class_zero=val_ignore_class_zero)
            callbacks = [validation] + callbacks

        if not no_im:
            # Add save images cb
            callbacks.append(SavePredictionImages(train, val))

        # Callback for plotting learning curves
        callbacks.append(LearningCurve())

        # Get FGBatchBalancer callbacks, etc.
        if hasattr(train, "n_fg_slices"):
            FGbalancer = FGBatchBalancer(train, logger=self.logger)
            callbacks = callbacks + [FGbalancer]
        callbacks = callbacks + [DividerLine(self.logger)]

        # Get initialized callback objects
        callbacks, cb_dict = init_callback_objects(callbacks, self.logger)

        # If ModelCheckPointClean is used, set the original model to store
        # the correct weights when using multi-GPU models
        cb = cb_dict.get("ModelCheckPointClean")
        if cb:
            cb.org_model = self.org_model

        # Fit the model
        # is_queued = bool(train.image_pair_loader.queue)
        self.logger.active_log_file = "training"
        self.logger.print_calling_method = False

        fit_kwargs = {
            "generator": train,
            "steps_per_epoch": train_steps,
            "epochs": n_epochs,
            "verbose": verbose,
            "callbacks": callbacks,
            "initial_epoch": init_epoch,
            "use_multiprocessing": use_multiprocessing,
            "workers": min(7, cpu_count()-1),
            "max_queue_size": 25,
            "shuffle": False
        }
        self.model.fit_generator(**fit_kwargs)
