import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
import argparse


# define hyperparameters from args

# optimizer momentum parameters
beta_1 = 0.5 
beta_2 = 0.999


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--Device", type=str, help = "Training device: Cuda:n/CPU")
parser.add_argument("-lr", "--LRate", type=int, help = "Learning Rate: Default .001")
parser.add_argument("-z", "--ZDim", type=int, help = "Noise Vector Size: Default 128")
parser.add_argument("-e", "--Epochs", type=int, help = "Number of Training epochs: Default 25")
parser.add_argument("-gp", "--Gradpenalty", type=int, help = "Coefficient of gradient penalty: Default 10")
parser.add_argument("-ds", "--DisplayStep", type=int, help = "Number of iterations between loss updates: Default 10")
parser.add_argument("-bs", "--BatchSize", type=int, help = "Batch Size per training iteration: Default 2")
args = parser.parse_args()

update = lambda v, a: a if a != None else v

device = update("cuda:0", args.Device)
z_dim = update(128, args.ZDim)
display_step = update(10, args.DisplayStep)
batch_size = update(2, args.BatchSize)
lr = update(.001, args.LRate)
epochs = update(25, args.Epochs)
c_lambda = update(10, args.Gradpenalty) #coefficient of gradient penalty

h_params = {
    'device' : device,
    'z_dim' : z_dim,
    'display_step' : display_step,
    'batch_size' : batch_size,
    'learning rate' : lr,
    'epochs' : epochs,
    'Coefficient of gradient penalty' : c_lambda,
    'Optimizer momentums' : (beta_1, beta_2)
}

def print_hyperparameters():
    print("Training hyperparameters:")
    for param in h_params.keys():
        print(f"{param} : {h_params[param]}")
    print("----------------------\nTraining started")


# create noise vector -- drawn from normal dist
def get_noise(n_samples, z_dim, device, im_chan=1):
    return torch.randn(n_samples, z_dim*im_chan, device=device)

# define generator
class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=32):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 2, kernel_size=(6,6,5)),
            self.make_gen_block(hidden_dim * 2, hidden_dim * 2, kernel_size=(5,5,3)),
            self.make_gen_block(hidden_dim * 2, hidden_dim*4, kernel_size=(5,5,3)),
            self.make_gen_block(hidden_dim * 4, hidden_dim*2, kernel_size=(7,7,5)),
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=5),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=(2,2,3), final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose3d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm3d(output_channels),
                nn.ReLU()
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose3d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
            )
    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

class Discriminator(nn.Module):

    def __init__(self, im_chan=1, hidden_dim=8):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim, kernel_size=(2,2,3)),
            self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size=5),
            self.make_disc_block(hidden_dim*2, hidden_dim * 4, kernel_size=(7,7,5)),
            self.make_disc_block(hidden_dim*4, hidden_dim * 2, kernel_size=(5,5,3)),
            self.make_disc_block(hidden_dim*2, hidden_dim * 2, kernel_size=(5,5,3)),
            self.make_disc_block(hidden_dim * 2, 1, kernel_size=(6,6,5), final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.Conv3d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm3d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv3d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):

        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)




# initialize gen, disc, and optimizer 
gen = Generator(z_dim, im_chan=1, hidden_dim=16).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator(im_chan=1, hidden_dim=8).to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

# initialize network weights -- W ~ N(0,.02^2)
def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)


def get_gradient(disc, real, fake, epsilon):

    # interpolate real and fake samples
    mixed_images = torch.add(real * epsilon, fake * (1 - epsilon))
    #print(mixed_images.shape)
    mixed_scores = disc(mixed_images)
    
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - torch.ones_like(gradient_norm))**2)
    return penalty

def get_gen_loss(disc_fake_pred):
    gen_loss = -torch.mean(disc_fake_pred)
    return gen_loss

def get_disc_loss(disc_fake_pred, disc_real_pred, gp, c_lambda):
    disc_loss = torch.mean(disc_fake_pred) + torch.mean(gp)*c_lambda - torch.mean(disc_real_pred)
    return disc_loss



# define brain dataset
class BrainDataset(Dataset):
    def __init__(self, scan_type="T2w"):
        self.scan_type=scan_type
        self.path = os.path.join(os.path.dirname(os.getcwd()), "research_dataset", "ds", self.scan_type)
        self.file_names = os.listdir(self.path)
        self.length = len(self.file_names)

    def reshape_samples(self, x):
        x = x[:,:,67:-20] # slice z down to 203
        p1, p2 = np.zeros((36, 290, 203)), np.zeros((37, 290, 203))
        x = np.concatenate((p1, x, p2), axis=0) # pad x up to 290 by evenly concatinating zero 290*203 tensors
        return x

    def path_to_array(self, path):
        image_array = nib.load(path)
        image_array = image_array.get_fdata()
        return np.array(image_array)

    def __getitem__(self, index):
        item_path = os.path.join(self.path, self.file_names[index])
        item = self.path_to_array(item_path)
        if item.shape != (290, 290, 203):
            item = self.reshape_samples(item)
        if self.scan_type=="T2w":
            item = item/88
        else:
            item = item/3    
        return item

    def __len__(self):
        return self.length


dataloader = DataLoader(BrainDataset(), batch_size=batch_size, shuffle=True, num_workers=4)


def train():
    dr = 5
    gr = 10
    disc_repeats, gen_repeats = 10, 1

    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    # training loop
    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            cur_batch_size = len(batch)
            real = batch.view((cur_batch_size, 1, 290, 290, 203)).to(device, dtype=torch.float)
            #print(real.shape)


            for _ in range(disc_repeats):
                ## Update discriminator ##
                disc_opt.zero_grad()
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                disc_fake_pred = disc(fake.detach())
                disc_real_pred = disc(real)

                # interpolate real and fake with random factor
                epsilon = torch.rand(len(real), 1, 1, 1, 1, device=device, requires_grad=True)
                grad = get_gradient(disc, real, fake.detach(), epsilon)
                gp = gradient_penalty(grad)
                disc_loss = get_disc_loss(disc_fake_pred, disc_real_pred, gp, c_lambda)

                # Update gradients
                disc_loss.backward(retain_graph=True)
                # Update optimizer
                disc_opt.step()

            ## Update generator ##
            for _ in range(gen_repeats):
                gen_opt.zero_grad()
                fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
                fake_2 = gen(fake_noise_2)
                disc_fake_pred = disc(fake_2)

                gen_loss =get_gen_loss(disc_fake_pred)

                gen_loss.backward()
                gen_opt.step()
            

            # Keep track of the average disc and gen loss
            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            # dynamically adjust train cycles to gen or disc domination
            if mean_discriminator_loss > mean_generator_loss:
                disc_repeats = dr
                gen_repeats = 1
            else:
                disc_repeats = 1
                gen_repeats = gr

            if cur_step % display_step == 0:
                print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            cur_step += 1
        print("epoch complete")
        torch.save(gen.state_dict(), 'models/gen_e{}.pt'.format(epoch))
        torch.save(disc.state_dict(), 'models/disc_e{}.pt'.format(epoch))
        print("gen and disc saved")

    print("training complete")
    torch.save(gen.state_dict(), 'models/gen.pt')
    torch.save(disc.state_dict(), 'models/disc.pt')
    print("gen and disc saved")

if __name__ == "__main__":
    print_hyperparameters()
    train()