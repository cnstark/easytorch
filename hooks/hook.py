class Hook:
    """
    Train:
    [before_run]
    for in train_epoch
        [before_train_epoch]
        for in train iters
            [before_train_iter]
            train iter
            [after_train_iter] ---> Iter  Val: val every n iters/inner_iters
        [after_train_epoch] ------> Epoch Val: val every n epoch
                                    [before_val]
                                    for in val iters
                                        [before_val_iter]
                                        val iter
                                        [after_val_iter]
                                    [after_val]
    [after_run]
    """

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_train_epoch(self, runner):
        pass

    def after_train_epoch(self, runner):
        pass

    def before_train_iter(self, runner):
        pass

    def after_train_iter(self, runner):
        pass

    def before_val(self, runner):
        pass

    def after_val(self, runner):
        pass

    def before_val_iter(self, runner):
        pass

    def after_val_iter(self, runner):
        pass

    def every_n_epochs(self, runner, n: int) -> bool:
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n: int) -> bool:
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n: int) -> bool:
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner) -> bool:
        return runner.inner_iter + 1 == len(runner.data_loader)

    def is_last_epoch(self, runner) -> bool:
        return runner.epoch + 1 == runner._max_epochs

    def is_last_iter(self, runner) -> bool:
        return runner.iter + 1 == runner._max_iters
