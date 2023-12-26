import datetime
import json
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
import numpy as np


def work_with_cam(max_img_heigh, max_img_wight):  # (local_dir: str = None):
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
        height, width = img.shape[:2]

        if height > max_img_heigh or width > max_img_wight:
            scale_ratio_h, scale_ratio_w = max_img_heigh/height, max_img_wight/width
            if scale_ratio_h < scale_ratio_w:
                scale_ratio = scale_ratio_h
            else:
                scale_ratio = scale_ratio_w
            # #calculate the 50 percent of original dimensions
            width = int(width * scale_ratio)
            height = int(height * scale_ratio)
            # # dsize
            dsize = (width, height)  # resize image
            img = cv2.resize(img, dsize)


        # if local_dir:
        #     file_path = os.path.join(local_dir + unic_name)
        #     os.chdir(local_dir)
        # else:
        #     file_path = os.path.join(os.getcwd() + unic_name)

        # cv2.imwrite(unic_name, img)
        break

    # close the camera
    cap.release()
    # close all the opened windows
    cv2.destroyAllWindows()

    return img, height, width


def get_background_api(prompt, negative_prompt="", sampler_name="DPM++ 2M Karras", steps=20, cfg_scale=7.0,
                       denoising_strength=0.0, width=512, height=512):
    url = "http://127.0.0.1:7860"

    payload = {
      "prompt": prompt,
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
    image = r['images'][0]
    return image



def refactor_person_with_background_api(prompt, img_b64, file_name, restore_faces=True, negative_prompt="", sampler_name="DPM++ 2M Karras",
                                        steps=20, cfg_scale=7.0, denoising_strength=0.4, width=512, height=512):
    url = "http://127.0.0.1:7860"

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "init_images": [
                img_b64
              ],
        "sampler_name": sampler_name,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "denoising_strength": denoising_strength,
        "restore_faces": restore_faces
    }


    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
    r = response.json()
    print(r)
    image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    image.save("ref_" + file_name)



def getting_person_without_background(cam_img):
    output = remove(cam_img, alpha_matting_erode_size=5)
    return output


def superimposing_person_background(b64_background, array_person, local_dir=""):

    img1 = Image.open(io.BytesIO(base64.b64decode(b64_background)))
    img2 = Image.fromarray(array_person)
    img1.paste(img2, (0, 0), mask=img2)
    unic_name = (''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                 + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".png")
    # Saving the image
    img1.save(unic_name)
    im_file = io.BytesIO()
    img1.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)
    return im_b64, unic_name




prompt = ("goblin's face, made of ancient weathered rock, tall grass, moss, trees, roots, half-timbered houses and "
          "stone staircases on the rock, winding upwards, against the sky with clouds, hyperealistic, surrealism, "
          "complex concept art, atmospheric, cinematic, very attractive and fabulous")
prompt2 = "London,  the road between the houses, RAW photo, high quality,film grain, bokeh, professional"
# get_background_api(prompt, width=640, height=480)
# my_image = cv2.imread(r"C:\Users\Komp\Desktop\Clear_work_with_cam\pers.jpg")
# print(my_image.shape)


def full_work_with_photo(prompt: str, local_dir: str = "", cfg_scale: str = "7", denoising_strength: str ="0.7",
                         max_heigh=1024, max_wight=1024):
    cfg_scale, denoising_strength = float(cfg_scale), float(denoising_strength)
    cam_img, height_img, width_img = work_with_cam(max_heigh, max_wight)
    j_person = getting_person_without_background(cam_img)
    img_b64_background = get_background_api(prompt=prompt, cfg_scale=cfg_scale, width=width_img, height=height_img)

    img_pers_with_back, name = superimposing_person_background(img_b64_background, j_person, local_dir)
    refactor_person_with_background_api(prompt, img_b64_background, name, cfg_scale=cfg_scale, width=width_img,
                                        height=height_img)



# full_work_with_photo(prompt)
# get_background_api(prompt)
# getting_person_without_background()
# get_background_api(prompt)

# img = Image.open('pers2.png')
# im_file = io.BytesIO()
# img.save(im_file, format="PNG")
# im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
# im_b64 = base64.b64encode(im_bytes)
# print(im_b64)



# im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
# img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

# with open("output10.png", "rb") as f:
#     # print(f.read())
#     im_b64 = base64.b64encode(f.read())
#     im_bytes = base64.b64decode(im_b64)
#     print(im_bytes)



# input = cv2.imread("so.jpg")
# scale_percent = 50
# #calculate the 50 percent of original dimensions
# width = int(input.shape[1] * scale_percent / 100)
# height = int(input.shape[0] * scale_percent / 100)
# # dsize
# dsize = (width, height) # resize image
# output = cv2.resize(input, dsize)
# cv2.imwrite('os3.png', output)
# output = remove(output, alpha_matting_erode_size=15, alpha_matting_background_threshold=100)
# cv2.imwrite("so2.png", output)



# src = cv2.imread("so2.png")
# scale_percent = 50
# #calculate the 50 percent of original dimensions
# width = int(src.shape[1] * scale_percent / 100)
# height = int(src.shape[0] * scale_percent / 100)
# # dsize
# dsize = (width, height) # resize image
# output = cv2.resize(src, dsize)
# cv2.imwrite('os3.png',output)

# img1 = Image.open('so5.png')
# im_file = io.BytesIO()
# img1.save(im_file, format="PNG")
# im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
# im_b64 = str(base64.b64encode(im_bytes).decode())
#
#
# refactor_person_with_background_api(prompt2, im_b64, "so9.png", restore_faces=True, width=728, height=984)

# img1 = Image.open('02.png')
# img1.paste(img2, (0, 0), mask=img2)
#
#     # Saving the image
# img1.save('so6.png')
print(os.path.join('df', 'w.png'))






# def load_input_image(path):
#     with open(path, "rb") as file:
#         return base64.b64encode(file.read()).decode()
# img = load_input_image('pers_ready3.png')
#
# img = Image.open('pers_ready3.png')
# print(img)
# print(type(img))
# work_with_cam()
# img = cv2.imread('i1.png')
# print(img.shape[:2])


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