import os
from eye_tracking import logging

data_path = "C:\Users\Admin\Downloads\code\dataset\full_HUST_LEBW\test"

ID = 0

for index, entry in enumerate(os.listdir(data_path)):
    print(entry)
    entrydir = os.path.join(data_path, entry)
    
    for frame_id, image in enumerate(os.listdir(entrydir)):
        frame_path = os.path.join(entrydir, image)
        logging(str(index), frame_id, frame_path) #frame_id == time, #people_ID
                    # print(people_id.name)
                    
                    
