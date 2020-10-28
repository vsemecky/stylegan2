from moviepy.editor import *
import scipy
import dnnlib
import dnnlib.tflib as tflib
from datetime import datetime
import pickle
import numpy as np
import PIL.Image
from training.misc import create_image_grid


# Init
networks_cache = {}
tflib.init_tf()  # Init TensorFlow
imageio.plugins.ffmpeg.download()  # Download ffmpeg if not installed


#
# Loads pre-trained network from pkl file or URL
# Result is cached for future use.
#
def load_network(pkl):
    if pkl in networks_cache.keys():
        # Return network from cache
        return networks_cache[pkl]
    else:
        # Load network from pkl file and store to cache
        with open(pkl, 'rb') as stream:
            print("Loading neurals: {}".format(pkl))
            networks_cache[pkl] = pickle.load(stream, encoding='latin1')
            return networks_cache[pkl]


#
# Generates unified filename based on parameters
#
def get_filename(prefix="video", time=None, psi=None, seed=None, timestamp=True):
    file_name = ""
    if timestamp:
        file_name += datetime.now().strftime("%Y-%m-%d %H:%M")
    if prefix:
        file_name += prefix
    if seed:
        file_name += " - seed={}".format(seed)
    if time:
        file_name += " - {}sec".format(time)
    if psi:
        file_name += " - psi={:03d}".format(int(100 * psi))
    file_name += ".mp4"  # Append extension

    return file_name


tflib.init_tf()


# from https://colab.research.google.com/drive/1ShgW6wohEFQtqs_znMna3dzrcVoABKIH
def generate_zs_from_seeds(seeds, Gs):
    zs = []
    enam = enumerate(seeds)
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
        zs.append(z)
    return zs


def generate_latent_images(zs, truncation_psi, save_npy, prefix):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)

    for z_idx, z in enumerate(zs):
        if isinstance(z, list):
            z = np.array(z).reshape(1, 512)
        elif isinstance(z, np.ndarray):
            z.reshape(1, 512)
        print('Generating image for step %d/%d ...' % (z_idx, len(zs)))
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1)  # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
        images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('%s%05d.png' % (prefix, z_idx)))
        if save_npy:
            np.save(dnnlib.make_run_dir_path('%s%05d.npy' % (prefix, z_idx)), z)


def line_interpolate(zs, steps):
    out = []
    for i in range(len(zs) - 1):
        for index in range(steps):
            fraction = index / float(steps)
            out.append(zs[i + 1] * fraction + zs[i] * (1 - fraction))
    return out


#
# @see sasdfrom https://github.com/dvschultz/stylegan2
#
def latent_interpolation_clip(pkl, psi=0.5, mp4_fps=30, duration=60, seeds=[]):
    # Loading neurals
    _G, _D, Gs = load_network(pkl)

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    zs = generate_zs_from_seeds(seeds, Gs)

    num_frames = int(np.rint(duration * mp4_fps))
    number_of_steps = int(num_frames / (len(zs) - 1)) +1  # @todo Prejmenovat na num_steps nebo steps_count/frames_count
    points = line_interpolate(zs, number_of_steps)

    # Generate_latent_images()
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    # Frame generation func for moviepy
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))

        # TIP: Oddebugovat run_generator,  co mu sem leze, zejmena len(zx)
        z_idx = frame_idx
        z = points[z_idx]

        # Puvodni loop
        if isinstance(z, list):
            z = np.array(z).reshape(1, 512)
        elif isinstance(z, np.ndarray):
            z.reshape(1, 512)

        Gs_kwargs.truncation_psi = psi
        noise_rnd = np.random.RandomState(1)  # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
        images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]

        # @todo Zbavit se gridu, kdyz potrebujeme jen jeden obrazek
        images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        grid = create_image_grid(images, [1, 1]).transpose(1, 2, 0)  # HWC
        # if grid.shape[2] == 1:
        #     grid = grid.repeat(3, 2)  # grayscale => RGB
        return grid

    # Return clip
    clip = VideoClip(make_frame, duration=duration)
    return clip


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
    clip = VideoClip(make_frame, duration=time)
    return clip


#
# Generates psi comparison VideoClip
#
def psi_comparison_clip(
            pkl=None,
            grid=None,
            mp4_fps=30,
            time=60,  # Duration in seconds
            smoothing_sec=1.0,
            randomize_noise=False,
            seed=420):

    # Default grid of psi truncations
    if grid is None:
        grid = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]

    # Overwrite psi values with MoviePy clips
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            psi = grid[row][col]
            grid[row][col] = latent_walk_clip(
                pkl=pkl,
                mp4_fps=mp4_fps,
                psi=psi,
                time=time,
                smoothing_sec=smoothing_sec,
                randomize_noise=randomize_noise,
                seed=seed
            )

    return clips_array(grid)


#
# Generates psi comparison VideoClip
#
def seed_comparison_clip(
            pkl=None,
            grid=None, # Grid of seeds
            psi=0.5,  # Truncation psi
            mp4_fps=30,
            time=60,  # Duration in seconds
            smoothing_sec=1.0,
            randomize_noise=False):

    # Default grid of seeds
    if grid is None:
        grid = [
            [420, 128, 846, 1420, 1128, 1846],
            [542, 251, 355, 1542, 1251, 1355],
        ]

    # Overwrite psi values with MoviePy clips
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            seed = grid[row][col]
            grid[row][col] = latent_walk_clip(
                pkl=pkl,
                mp4_fps=mp4_fps,
                psi=psi,
                time=time,
                smoothing_sec=smoothing_sec,
                randomize_noise=randomize_noise,
                seed=seed
            )

    return clips_array(grid)