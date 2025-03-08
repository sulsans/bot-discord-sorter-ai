from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import h5py



def get_class(model_path, labels_path, image_path):

    f = h5py.File(model_path, mode="r+")
    model_config_string = f.attrs.get("model_config")
    if model_config_string.find('"groups": 1,') != -1:
        model_config_string = model_config_string.replace('"groups": 1,', '')
        f.attrs.modify('model_config', model_config_string)
        f.flush()
        model_config_string = f.attrs.get("model_config")
        assert model_config_string.find('"groups": 1,') == -1

    f.close()

    np.set_printoptions(suppress=True)
    model = load_model(model_path, compile=False)
    class_names = open(labels_path, "r", encoding="utf-8").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    if class_name[2:] == 'kertas\n':
        output = "ini adalah sampah kertas, kertas dapat didaur ulang dengan cara dibuat menjadi bubur kertas dan dijadikan kertas baru. kalau ingin dibuang tempat sampahnya berwarna biru. jika tidak dibuang dapat memerlukan waktu 2-6 minggu hingga beberapa bulan"

    elif class_name[2:] == 'alas kaki\n':
        output = "ini adalah sampah sepatu atau sendal sampah ini dapat didaur ulang material dengan cara memisahkan bagian bagian sepatu atau sendal berdasarkan material atau di daur ulang menjadi kerajinan. jika dibiarkan akan memerlukan waktu 25-80 tahun untuk terurai"

    elif class_name[2:] == 'plastik\n':
        output = "ini adalah sampah plastik sampah plastik dapat didaur ulang dengan cara dileburkan dan dibentuk ulang, jika ingin dibuang dibuang ke tempah samppah anorganik atau warna kuning. jika dibiarkan akan memerlukan waktu 10 hingga ribuan tahun untuk terurai"

    elif class_name[2:] == 'logam\n':
        output = "ini adalah sampah logam, sampah logam dapat didaur ulang dengan cara dileburkan dan dimurnikan, jika ingin dibuang, buanglah ke tempat sampah warna kuning atau anorganik. jika dibiarkan akan memerlukan waktu lebih dari 100 tahun untuk terurai"

    elif class_name[2:] == 'organik\n':
        output = "ini adalah sampah organik, sampah jenis ini dapaat didaur ulang menjadi kompos, jika ingin dibuang, dibang ketempat sampah berwarna hijau. jika dibiarkan akan terurai dalam hitungan hari atau minggu"

    return(output)