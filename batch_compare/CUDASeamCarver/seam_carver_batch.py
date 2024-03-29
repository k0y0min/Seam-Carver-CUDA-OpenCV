import numpy as np
import time
import os
from PIL import Image

IMAGE_DIRECTORY = "images"
seam_values = [4, 8, 12, 16, 20]
iteration_values = [10,20,30,40,50]
def compute_energy(image: np.ndarray):
    #forward energy
    R = np.roll(image, 1, axis=0) 
    L = np.roll(image, -1, axis=0)
    U = np.roll(image, 1, axis=1)
    D = np.roll(image, -1, axis=1)

    dx_sq = np.square(R-L)
    dy_sq = np.square(U-D)
    delta_sq = np.sum(dx_sq, axis = 2, dtype = np.double) + np.sum(dy_sq, axis = 2, dtype = np.double)
    energy = np.sqrt(delta_sq)

    #border pixels set to 1000
    energy[:,0] = 1000
    energy[0,:] = 1000
    energy[image.shape[0]-1, :] = 1000
    energy[:, image.shape[1]-1] = 1000

    return energy

def find_vertical_seam(image: np.ndarray, energy=None):
    if energy is None:
        energy = compute_energy(image)
        # Add a small random noise to the energy, which is generated by
        # an image assumed to contain integer values between [0, 255]
        # (thus may contain many duplicate values), to avoid variations
        # between implementations yielding different results.

        # Storing the internal random state for later reversion
        random_state = np.random.get_state()
        # Seeding the random state to 0
        np.random.seed(0)
        # generating the random noise
        noise = np.random.randn(*energy.shape) / (1000 * (image.size ** (0.5)))
        energy = energy + noise
        # Reverting the random state to what we started with
        np.random.set_state(random_state)

    r, c = energy.shape
    sol = np.zeros((r,c), dtype = int)
    cM = energy.copy()
    min_seam = np.zeros(r, dtype= int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                t = np.argmin(cM[i - 1, j:j + 2])
                sol[i, j] = t + j
                min_energy = cM[i - 1, t + j]
            else:
                t = np.argmin(cM[i - 1, j - 1:j + 2])
                sol[i, j] = t + (j - 1)
                min_energy = cM[i - 1, t + j - 1]

            cM[i, j] += min_energy
    
    j = np.argmin(cM[-1])
    for i in reversed(range(r)):
        min_seam[i] = j
        j = sol[i, j]
    
    return min_seam

def run_batch(num_seams: int, num_iterations: int):
    image_names = os.listdir(IMAGE_DIRECTORY)[:num_iterations]
    images = []
    for image_name in image_names:
        image_path = os.path.join(IMAGE_DIRECTORY, image_name)
        p = Image.open(image_path)
        p = p.convert(mode="RGB")
        image = np.array(p).astype(int)
        images.append(image)
    start_time = time.time()
    for image in images:
        for _ in range(min([num_seams, image.shape[1]])):
            vertical_indices = tuple(np.arange(image.shape[0]))
            horizontal_indices = tuple(find_vertical_seam(image))
            mask = np.full(image.shape, True, dtype=bool)
            mask[vertical_indices, horizontal_indices] = False
            image = image[mask].reshape((image.shape[0], image.shape[1] - 1, image.shape[2]))
        #final_image = Image.fromarray(image.astype(np.uint8))
        #final_image.save("carved_images/{}_{}_carved.png".format(num_seams, image))
    end_time = time.time()

    elapsed_time = end_time - start_time
    #elapsed_time = round(elapsed_time/60 , 2)
    print("{} iterations | {} seams | {} seconds".format(num_iterations, num_seams, elapsed_time)) 
                                                         
def main():
    # for seams in seam_values:
    #     for iterations in iteration_values:
    #         run_batch(seams, iterations)
    run_batch(8, 1)
if __name__ == "__main__":
    main()
