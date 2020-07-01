from CAE import CAE
from DataLoader import DataLoader
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    data_loader = DataLoader("src/data/char_image/")
    data, code_list = data_loader.get_data()
    print(len(data), flush=True)

    # cae = CAE()
    # cae.create_model(32)
    # cae.train(data, epochs=100, batch_size=32)
    # cae.save_autoencoder("src/data/model/autoencoder.h5")
    # cae.save_encoder("src/data/model/encoder.h5")

    cae = load_model("src/data/model/encoder.h5")
    pred = cae.predict(data)

    with open("src/data/visual_keyedvector.vec", "w", encoding="utf8") as f:
        f.write("{} {}\n".format(pred.shape[0], pred.shape[1]))
        for code, vec in zip(code_list, pred):
            write_data = [chr(code)]
            write_data += ["{:.6f}".format(v) for v in vec]
            f.write("{}\n".format(" ".join(write_data)))
