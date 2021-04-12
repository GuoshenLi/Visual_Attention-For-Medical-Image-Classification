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

    model.load_model()
    print("Testing...")

    _, _, _, _, acc1, acc2, acc, nr1, br1, ir1, kappa1, nr2, br2, ir2, kappa2, \
        nr, br, ir, kappa = model.test(batch_size=4)

    print ("Net1_accuracy:", acc1)
    print ("Net2_accuracy:", acc2)
    print ("Total_accuracy:", acc)

    print ("Net1_normal_recall:", nr1)
    print ("Net2_normal_recall:", nr2)
    print ("Total_normal_recall:", nr)

    print ("Net1_bleed_recall:", br1)
    print ("Net2_bleed_recall:", br2)
    print ("Total_bleed_recall:", br)

    print ("Net1_inflam_recall:", ir1)
    print ("Net2_inflam_recall:", ir2)
    print ("Total_inflam_recall:", ir)

    print ("Net1_kappa:", kappa1)
    print ("Net2_kappa:", kappa2)
    print ("Total_kappa:", kappa)