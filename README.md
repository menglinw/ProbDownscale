## Program Structure
* TaskExtractor
    * Description
        * take data and extract several tasks randomly 
        * each task including training and testing data
        * training and testing data should be flattened and ready for training
    * Input
        * data: read from netcdf4, 3d np.ndarray like
        * n_task: number of task
        * test_proportion: test set proportion within each task
    * Output
        * train_X
            * dim: (task, train_X)
        * train_Y
            * dim: (task, train_Y)
        * test_X
            * dim: (task, test_X)
        * test_Y
            * dim: (task, test_Y)

* ModelGenerator
    * output: a model without compiling

* MetaTrain

* Downscale

* Evaluation