## Program Structure
* TaskExtractor
    * Role Description
        * For meta training
            * extract several tasks randomly when calling **meta_next** method
            * each task including training and testing data
            * training and testing data should be flattened and ready for training
        * For downscaling
            * extract one task sequentially when calling **ds_next** method
            * each task including training and testing data (how to split?)
            * data should be flattened and ready for training
    * Input
        * data: read from netcdf4, 3d np.ndarray like
        * n_task: number of task
        * test_proportion: test set proportion within each task
        * meta train: 
            * true --> return batch of task randomly, train test split randomly
            * false --> return one task at a time until no left, train test split sequentially
    * Output
        * train_X
            * dim: (task, train_X)
        * train_Y
            * dim: (task, train_Y)
        * test_X
            * dim: (task, test_X)
        * test_Y
            * dim: (task, test_Y)
        * location_list
            * a list of location for each task [[lat, lon], ...]

* Downscaler
    * Role Description
        * take a task extractor, target model and loss
        * call **meta_train** for meta training to learn the initialization
        * call **downscale** for downscaling 

    * MetaTrain
        * Input
            * model: model without compiling
            * loss function: loss function for the model
            * meta_step: training steps of meta training
            * meta_optimizer: optimizer of meta training
            * inner_step: training steps of inner training
            * inner_optimizer: optimizer of inner training
            * task extractor: object defined above
        * Output
            * model with optimized initialization
    * Downscale
        * Description: 
            * using optimized initialization to train a group of models
            * one model for each grid
            * predict using these models 
            * save downscaled data and true data
        * Input: 
            * model with optimized initialization
            * task extractor
        * Output:
            * downscaled data and true test data