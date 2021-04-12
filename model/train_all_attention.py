from models.AGDN_nonlocal import DenseNet


if __name__ == '__main__':


    # some default params dataset/architecture related
    train_params  = {
    'batch_size': 16,
    'n_epochs': 500,
    'initial_learning_rate': 0.001,
    'reduce_lr_epoch_1': 150,  # epochs * 0.1
    'reduce_lr_epoch_2': 300,  # epochs * 0.1
    'validation_set': True
    }

    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    print("Initialize the model..")
    model = DenseNet()

    # model.load_model()
    model.train_all_epochs(train_params)

