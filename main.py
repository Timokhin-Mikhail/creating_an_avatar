import datetime
import string
import random
import os
import cv2
import time
import requests
import io
import base64
from rembg import remove
from PIL import Image


def work_with_cam(local_dir: str = None):
    # SET THE COUNTDOWN TIMER for simplicity we set it to 3 We can also take this as input
    TIMER = int(5)

    # Open the camera
    cap = cv2.VideoCapture(0)
    while True:
        prev = time.time()
        font = cv2.FONT_HERSHEY_SIMPLEX
        while TIMER >= 0:
            ret, img = cap.read()

            # Display countdown on each frame
            # specify the font and draw the
            # countdown using puttext
            cv2.putText(img, str(TIMER),
                        (250, 250), font,
                        8, (0, 255, 255),
                        8, cv2.LINE_AA)
            cv2.imshow('cam', img)
            cv2.waitKey(125)

            # current time
            cur = time.time()

            # Update and keep track of Countdown
            # if time elapsed is one second
            # then decrease the counter
            if cur - prev >= 1:
                prev = cur
                TIMER = TIMER - 1

        ret, img = cap.read()
        cv2.putText(img, "",
                    (250, 250), font,
                    8, (0, 255, 255),
                    8, cv2.LINE_AA)
        cv2.imshow('cam', img)
        cv2.waitKey(2000)

        unic_name = (''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                     + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".png")
        if local_dir:
            file_path = os.path.join(local_dir + unic_name)
            os.chdir(local_dir)
        else:
            file_path = os.path.join(os.getcwd() + unic_name)
        # height, width = img.shape[:2]
        cv2.imwrite(unic_name, img)
        break

    # close the camera
    cap.release()
    # close all the opened windows
    cv2.destroyAllWindows()
    return file_path  # , height, width

def get_background_api(propmt, negative_prompt="",sampler_name="DPM++ 2M Karras", steps=20, cfg_scale=7,
                       denoising_strength =0, width=512, height=512):
    url = "http://127.0.0.1:7860"

    payload = {
      "prompt": propmt,
      "negative_prompt": negative_prompt,
      "sampler_name": sampler_name,
      "steps": steps,
      "cfg_scale": cfg_scale,
      "width": width,
      "height": height,
      "denoising_strength": denoising_strength,

    }
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    r = response.json()

    image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    image.save('output10.png')


def refactor_person_with_background_api(propmt, img_like_str, negative_prompt="", sampler_name="DPM++ 2M Karras",
                                        steps=20, cfg_scale=7, denoising_strength=0.4, width=512, height=512):
    url = "http://127.0.0.1:7860"

    payload = {
        "prompt": propmt,
        "negative_prompt": negative_prompt,
        "init_images": [
                img_like_str
              ],
        "sampler_name": sampler_name,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "denoising_strength": denoising_strength,
    }
    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)

    r = response.json()

    image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    image.save('output10.png')



def getting_person_without_background():
    input_path = 'Y6R7QJLVBO23-10-2023_18-32-06.jpg'
    output_path = 'pers7.png'
    input = cv2.imread(input_path)
    output = remove(input, alpha_matting_erode_size=5)
    cv2.imwrite(output_path, output)


def superimposing_person_background():

    img1 = Image.open('output1.png')
    # Opening the secondary image (overlay image)
    img2 = Image.open('pers7.png')
    # Pasting img2 image on top of img1
    # starting at coordinates (0, 0)
    img1.paste(img2, (0, 0), mask=img2)
    # Saving the image
    img1.save('pers_ready3.png')




prompt = ("goblin's face, made of ancient weathered rock, tall grass, moss, trees, roots, half-timbered houses and "
          "stone staircases on the rock, winding upwards, against the sky with clouds, hyperealistic, surrealism, "
          "complex concept art, atmospheric, cinematic, very attractive and fabulous")
# get_background_api(prompt, width=640, height=480)
# my_image = cv2.imread(r"C:\Users\Komp\Desktop\Clear_work_with_cam\pers.jpg")
# print(my_image.shape)



# def load_input_image(path):
#     with open(path, "rb") as file:
#         return base64.b64encode(file.read()).decode()
# img = load_input_image('pers_ready3.png')
#
# img = Image.open('pers_ready3.png')
# print(img)
# print(type(img))
# work_with_cam()
img = cv2.imread('i1.png')
print(img)
print(type(img))

# work_with_cam()

# img1 = Image.open('pers_ready2.png')
#
# image_64_encode = base64.b64encode(img1.tobytes()).decode()
# print(image_64_encode)
# print("\n============\n==============\n")
# print(str(image_64_encode))

# refactor_person_with_background_api(prompt, load_input_image('pers_ready3.png'), width=640, height=480)



# work_with_cam(r"C:\Users\Komp\Desktop\cam_prog\per")
# print(os.getcwd())
# {
#   "prompt": "",
#   "negative_prompt": "",
#   "styles": [
#     "string"
#   ],
#   "seed": -1,
#   "subseed": -1,
#   "subseed_strength": 0,
#   "sampler_name": "string",
#   "steps": 50,
#   "cfg_scale": 7,
#   "width": 512,
#   "height": 512,
#   "restore_faces": true,
#   "tiling": true,
#   "do_not_save_samples": false,
#   "do_not_save_grid": false,
#   "eta": 0,
#   "denoising_strength": 0.75,
#   "override_settings": {},
#   "override_settings_restore_afterwards": true,
#   "refiner_checkpoint": "string",
#   "refiner_switch_at": 0,
#   "disable_extra_networks": false,
#   "comments": {},
#   "init_images": [
#     "string"
#   ],
#   "resize_mode": 0,
#   "image_cfg_scale": 0,
#   "mask": "string",
#   "mask_blur_x": 4,
#   "mask_blur_y": 4,
#   "mask_blur": 0,
#   "inpainting_fill": 0,
#   "inpaint_full_res": true,
#   "inpaint_full_res_padding": 0,
#   "inpainting_mask_invert": 0,
#   "initial_noise_multiplier": 0,
#   "latent_mask": "string",
#   "sampler_index": "Euler",
#   "include_init_images": false,
#   "alwayson_scripts": {}
# }
#
# {
#   "prompt": "",
#   "negative_prompt": "",
#   "styles": [
#     "string"
#   ],
#   "seed": -1,
#   "subseed": -1,
#   "subseed_strength": 0,
#   "sampler_name": "string",
#   "steps": 50,
#   "cfg_scale": 7,
#   "width": 512,
#   "height": 512,
#   "restore_faces": true,
#   "tiling": true,
#   "do_not_save_samples": false,
#   "do_not_save_grid": false,
#   "eta": 0,
#   "denoising_strength": 0,
#   "override_settings": {},
#   "override_settings_restore_afterwards": true,
#   "refiner_checkpoint": "string",
#   "refiner_switch_at": 0,
#   "disable_extra_networks": false,
#   "comments": {},
#   "enable_hr": false,
#   "firstphase_width": 0,
#   "firstphase_height": 0,
#   "hr_scale": 2,
#   "hr_upscaler": "string",
#   "hr_second_pass_steps": 0,
#   "hr_resize_x": 0,
#   "hr_resize_y": 0,
#   "hr_checkpoint_name": "string",
#   "hr_sampler_name": "string",
#   "hr_prompt": "",
#   "hr_negative_prompt": "",
#   "sampler_index": "Euler",
#   "alwayson_scripts": {}
# }