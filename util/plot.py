# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 20:25:41 2022

@author: User
"""
import matplotlib.pyplot as plt




def plot_images(images,outputs):
    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=15, sharex=True, sharey=True, figsize=(25,4))
    
    # input images on top row, reconstructions on bottom
    for images, row in zip([images, outputs], axes):
        #print(len(images))
        #print(len(row))
        for img, ax in zip(images, row):
            #print(img)
            #print(np.shape(img))
            #print(np.shape(np.squeeze(img)))
            #img = img[-1::]
            
            #unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            #unorm(img)
            img = img[:,:,::-1].transpose((2,1,0))
            #print(np.shape(img))
            #print(np.shape(np.squeeze(img)))
            #ax.imshow(np.squeeze(img), cmap='gray')
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

def plot_loss_distribution(SHOW_MAX_NUM,positive_loss,defeat_loss):
    # Importing packages
    import matplotlib.pyplot as plt2
    # Define data values
    x = [i for i in range(SHOW_MAX_NUM)]
    y = positive_loss
    z = defeat_loss
    print(x)
    print(positive_loss)
    print(defeat_loss)
    # Plot a simple line chart
    #plt2.plot(x, y)
    # Plot another line on the same chart/graph
    #plt2.plot(x, z)
    plt2.scatter(x,y,s=1)
    plt2.scatter(x,z,s=1) 
    plt2.show()