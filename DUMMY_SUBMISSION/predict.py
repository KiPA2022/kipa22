from Demo_densebiasnet import train_net, predict


if __name__ == '__main__':
    net_S = train_net(
        checkpoint_dir='weights',
        is_load=True,
        load_epoch=200,
        is_train=False
    )
    # Read images from /input, write predictions to /output (leave model name to blank)
    predict(net_S, save_path='/output', img_path='/input', model_name='')
