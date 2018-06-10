import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
print(sys.path)
import json
from keras_audio.library.cifar10 import Cifar10AudioClassifier
from keras_audio.library.utility.gtzan_loader import download_gtzan_genres_if_not_found


def load_audio_path_label_pairs():
    with open('songid_list.json') as jf:
        songid_list = json.load(jf)
    with open('data_dict_album.json') as jf:
        data_dict = json.load(jf)
    list_gen = ['metal', 'rock', 'dance', 'hiphop', 'classical', 'reggae', 'jazz', 'folk']
    class_map = dict(zip(list_gen, range(0, len(list_gen))))
    pairs = []
    for sid in songid_list:
        audio_path = os.path.join('/newvolume/full_spectrogram_dimexp/', str(sid)+'.npy')
        genre = data_dict.get(str(sid))['genre']
        pairs.append((audio_path, class_map[genre]))
    return pairs


def main():
    audio_path_label_pairs = load_audio_path_label_pairs()
    print('loaded: ', len(audio_path_label_pairs))

    classifier = Cifar10AudioClassifier()
    batch_size = 128
    epochs = 200
    history = classifier.fit(audio_path_label_pairs, model_dir_path='/newvolume/keras-audio-master/demo/models', batch_size=batch_size, epochs=epochs)


if __name__ == '__main__':
    main()
