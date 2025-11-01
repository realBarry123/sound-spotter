from data_utils import *
import csv

def trim_leading_space(input_wav: torch.Tensor, threshold: float) -> torch.Tensor:
    pass


def sort_sounds():
    #pass

    file_name_cat = {}
    categories = set()

    filepath = 'ESC-50-master\ESC-50-master\meta\esc50.csv'
    csv_file = open(filepath, 'r', newline = '')
    with csv_file as csvfile:
        csv_reader = csv.reader(csvfile)

        for row in csv_reader:
            key = row[0]
            val = row[3]

            file_name_cat[key] = val

            categories.add(val)

    print(categories)
    
    """{'clapping', 'breathing', 'category', 'insects', 'thunderstorm', 'church_bells',
      'hen', 'chirping_birds', 'fireworks', 'clock_alarm', 'sheep', 'water_drops',
        'snoring', 'car_horn', 'chainsaw', 'toilet_flush', 'pig', 'clock_tick',
        'crying_baby', 'sea_waves', 'airplane', 'siren', 'sneezing', 'coughing',
        'washing_machine', 'dog', 'can_opening', 'wind', 'engine', 'frog', 'door_wood_creaks',
        'rooster', 'footsteps', 'crackling_fire', 'pouring_water', 'cat', 'drinking_sipping',
        'helicopter', 'door_wood_knock', 'train', 'rain', 'glass_breaking', 'mouse_click',
        'hand_saw', 'cow', 'crickets', 'vacuum_cleaner', 'laughing', 'brushing_teeth', 'crow',
        'keyboard_typing'}
    """



sort_sounds()
    

