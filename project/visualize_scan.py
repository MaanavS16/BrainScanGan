import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class visualizeScan:
    def __init__(self, path, scan_type, cmap='gray'):
        self.path = path
        self.scan_type = scan_type
        self.cmap = cmap
    
    def path_to_array(self):
        image_array = nib.load(self.path)
        image_array.get_fdata()
        image_array = image_array.get_fdata()
        #image_array.shape
        return np.array(image_array)

    def show_scan_slices(self, x=100):
        plt.figure()
        f, axarr = plt.subplots(1,3) 
        plt.title("Brain Scan Slices")

        item = self.path_to_array()

        axarr[0].imshow(item[100], cmap=self.cmap)
        axarr[0].title.set_text("Slice with fixed X")
        axarr[1].imshow(item[:,100], cmap=self.cmap)
        axarr[1].title.set_text("Slice with fixed Y")
        axarr[2].imshow(item[:,:,100], cmap=self.cmap)
        axarr[2].title.set_text("Slice with fixed Z")

        plt.show()


    def show_slice_anim(self, frames=202):
        fig, ax = plt.subplots()
        item = self.path_to_array()

        ims = []
        for i in range(frames):
            im = ax.imshow(item[:,:,i], animated=True)
            if i == 0:
                ax.imshow(item[:,:,i])
            ims.append([im])
        
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)
        plt.show()

