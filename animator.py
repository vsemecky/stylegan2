import pickle
import numpy as np
import imageio
import moviepy.editor
import scipy
import dnnlib.tflib as tflib

from training.misc import create_image_grid

# Init
networks_cache = {}
tflib.init_tf()  # Init TensorFlow
imageio.plugins.ffmpeg.download()  # Download ffmpeg if not installed


#
# Loads pretrained network from pkl. Result is cached for future use.
#
def load_network(pkl):
    if pkl in networks_cache.keys():
        # Return network from cache
        print("Cached neurals: {}".format(pkl))
        return networks_cache[pkl]
    else:
        # Load network from pkl file and store to cache
        with open(pkl, 'rb') as stream:
            print("Loading neurals: {}".format(pkl))
            networks_cache[pkl] = pickle.load(stream, encoding='latin1')
            return networks_cache[pkl]


#
# Generates latent walk VideoClip
#
def latent_walk_clip(
            pkl=None,
            mp4_fps=30,
            psi=0.5,  # Truncation psi
            time=60,  # Duration in seconds
            smoothing_sec=1.0,
            randomize_noise=False,
            seed=420):

    # Nepouzivane parametry z puvodni funkce
    grid_size=[1, 1]
    image_shrink=1
    image_zoom=1

    # Nacist neuronku
    # with open(pkl, 'rb') as stream:
    #     _G, _D, Gs = pickle.load(stream, encoding='latin1')

    # Nacist neuronku @todo Zkontrolovat jestli se to cachuje
    tflib.init_tf()  # @todo Musi se to tady inicializovat? Nestaci to co je na globalni urovni?
    _G, _D, Gs = load_network(pkl)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    num_frames = int(np.rint(time * mp4_fps))
    random_state = np.random.RandomState(seed)

    # Generating latent vectors
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:]  # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, None, truncation_psi=psi, randomize_noise=randomize_noise, output_transform=fmt)

        images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        grid = create_image_grid(images, grid_size).transpose(1, 2, 0)  # HWC
        # if image_zoom > 1:
        #     grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        # if grid.shape[2] == 1:
        #     grid = grid.repeat(3, 2)  # grayscale => RGB
        return grid

    # Return clip
    clip = moviepy.editor.VideoClip(make_frame, duration=time)
    return clip

