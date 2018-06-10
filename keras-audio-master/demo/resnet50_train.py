import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
print(sys.path)
from keras_audio.library.resnet50 import ResNet50AudioClassifier
from keras_audio.library.utility.gtzan_loader import download_gtzan_genres_if_not_found
import json

def load_audio_path_label_pairs():
    with open('songid_list.json') as jf:
        songid_list = json.load(jf)
    with open('data_dict_album.json') as jf:
        data_dict = json.load(jf)
    list_gen = ['metal', 'rock', 'dance', 'hiphop', 'classical', 'reggae', 'jazz', 'folk']
    class_map = dict(zip(list_gen, range(0, len(list_gen))))
    pairs = []
    for sid in songid_list:
        audio_path = os.path.join('/homedtic/vbajpai/masters/refined_data_spectrogram/', str(sid)+'.npy')
        genre = data_dict.get(str(sid))['genre']
        pairs.append((audio_path, class_map[genre]))
    return pairs


def main():
    audio_path_label_pairs = load_audio_path_label_pairs()
    print('loaded: ', len(audio_path_label_pairs))

    classifier = ResNet50AudioClassifier()
    batch_size = 2
    epochs = 40
    history = classifier.fit(audio_path_label_pairs, model_dir_path='./models', batch_size=batch_size, epochs=epochs)


if __name__ == '__main__':
    main()
