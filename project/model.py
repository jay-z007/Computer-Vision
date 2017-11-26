import torch
import torch.nn as nn


## TODO: Change the models to include text embeddings
## TODO: Add FC to reduce the text_embedding to the size of nt
class _netG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, nte, nt):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + nt, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.encode_text = nn.Sequential(
            nn.Linear(nte, nt), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

## TODO: pass nt and text_embedding size to the G and D and add FC to reduce text_embedding_size to nt
class _netD(nn.Module):
    def __init__(self, ngpu, nc, ndf, nte, nt):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True))
        # state size. (ndf*8) x 4 x 4
        ## add another sequential plot after this line to add the embedding and process it to find a single ans

        self.encode_text = nn.Sequential(
            nn.Linear(nte, nt),
            nn.LeakyReLU(0.2, inplace=True)

        )

        self.concat_image_n_text = nn.Sequential(
            nn.Conv2d(ndf * 8 + nt, ndf * 8, 4, 1, 0, bias=False), ## TODO: Might want to change the kernel size and stride
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, txt_embedding):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            encoded_img = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
            encoded_text = nn.parallel.data_parallel(self.encode_text, txt_embedding, range(self.ngpu))
            ## add the same things as in the else part
        else:
            encoded_img = self.main(input)
            encoded_text = self.encode_text(txt_embedding)
            encoded_text = encoded_text.expand(encoded_text.size(), 1, 1)
            encoded_text = encoded_text.repeat(1, 4, 4) ## can also directly expand, look into the syntax
            output = self.concat_image_n_text(torch.cat((encoded_img, encoded_text)), 0)

        return output.view(-1, 1).squeeze(1)
