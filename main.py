import datetime
import json
import string
import random
import cv2
import time
import requests
import io
import base64
from rembg import remove
from PIL import Image



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
        break

    # close the camera
    cap.release()
    # close all the opened windows
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, height, width


def getting_person_without_background(cam_img):
    output = remove(cam_img, alpha_matting_erode_size=5)
    cv2.imwrite("im2.png", output)
    return output


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


def superimposing_person_background(b64_background, array_person):

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
    im_b64 = base64.b64encode(im_bytes).decode()
    return im_b64, unic_name


def refactor_person_with_background_api(prompt, img_b64, file_name, restore_faces=True, negative_prompt="", sampler_name="DPM++ 2M Karras",
                                        steps=20, cfg_scale=7.0, denoising_strength=0.4, width=512, height=512):
    url = "http://127.0.0.1:7860"

    prompt += ", beautiful person, beautiful male face"

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
    image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    image.save("ref_" + file_name)


def full_work_with_photo(prompt: str, local_dir: str = "", cfg_scale: str = "7", denoising_strength: str ="0.45",
                         max_heigh=1024, max_wight=1024):
    cfg_scale, denoising_strength = float(cfg_scale), float(denoising_strength)
    cam_img, height_img, width_img = work_with_cam(max_heigh, max_wight)
    j_person = getting_person_without_background(cam_img)
    img_b64_background = get_background_api(prompt=prompt, cfg_scale=cfg_scale, width=width_img, height=height_img)

    img_pers_with_back, name = superimposing_person_background(img_b64_background, j_person, local_dir)
    refactor_person_with_background_api(prompt, img_pers_with_back, name, cfg_scale=cfg_scale, width=width_img,
                                        height=height_img)


prompt = ("goblin's face, made of ancient weathered rock, tall grass, moss, trees, roots, half-timbered houses and "
          "stone staircases on the rock, winding upwards, against the sky with clouds, hyperealistic, surrealism, "
          "complex concept art, atmospheric, cinematic, very attractive and fabulous")
prompt2 = "extremely detailed CG, perfect lighting, 8k wallpaper, RAW photography, masterpiece: 1.4, realistic, HDR, wallpaper, huge png, atlantistech,cityscape,(((crashed big spaceship overgrown with vines))), ruins, bioluminescent, blue background, cyberpunk style,realistic lighting, dark sci-fi movie, 8k, sharpness, focus, cinematic frame, volumetric light (ink splashes: 1), (color splashes: 1), (colorful: 1), (watercolor: 1.2), soft lighting, stella monument in the city center octane rendering, Unreal Engine, 3D rendering, best quality, UHD, 8K"


if __name__ == "__main__":
    full_work_with_photo(prompt2, cfg_scale="4", denoising_strength="0.3")

