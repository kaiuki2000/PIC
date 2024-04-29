import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt

def sigBkgEff(myModel, X_test, y_test, returnDisc=False, fc=0.07):

    '''
    Given a model, make the histograms of the model outputs to get the ROC curves.

    Input:
        myModel: A keras model
        X_test: Model inputs of the test set
        y_test: Truth labels for the test set
        returnDisc: If True, also return the raw discriminant 
        fc: The amount by which to weight the c-jet prob in the disc. The
            default value of 0.07 corresponds to the fraction of c-jet bkg
            in ttbar.

    Output:
        effs: A list with 3 entries for the l, c, and b effs
        disc: b-tagging discriminant (will only be returned if returnDisc is True)
    '''

    # Evaluate the performance with the ROC curves!
    predictions = myModel.predict(X_test,verbose=True)

    # To make sure you're not discarding the b-values with high
    # discriminant values that you're good at classifying, use the
    # max from the distribution
    disc = np.log(np.divide(predictions[:,2], fc*predictions[:,1] + (1 - fc) * predictions[:,0]))
    
    '''
    Note: For jets w/o any tracks
    '''
    
    discMax = np.max(disc)
    discMin = np.min(disc)
    
    myRange=(discMin,discMax)
    nBins = 200

    effs = []
    plt.figure()
    for output, flavor in zip([0,1,2], ['l','c','b']):

        ix = (np.argmax(y_test,axis=-1) == output)
        
        # Plot the discriminant output
        nEntries, edges ,_ = plt.hist(disc[ix],alpha=0.5,label='{}-jets'.format(flavor),
                                      bins=nBins, range=myRange, density=True, log=True)

        '''
        nEntries is just a sum of the weight of each bin in the histogram.
        
        
        Since high Db scores correspond to more b-like jets, compute the cummulative density function
        from summing from high to low values, this is why we reverse the order of the bins in nEntries
        using the "::-1" numpy indexing.
        '''
        
        # IT TOOK ME A LONG TIME TO UNDERSTAND THIS, so I'll write it down in case I need to come back later and have 
        # forgotten how this works...
        # So, it's easier to understand if we think in terms of the efficency. A b-efficiency of 1, means that we tag all 
        # b-jets as b-jets.
        # An efficiency of 0, means that we don't tag any b-jet as a b-jet. If we were to increase the efficiency by just 
        # a little bit, we would tag as 
        # b-jets a very small amount of b-jets. These would naturally be the ones that we are the most sure correspond to
        # actual b-jets, so the ones
        # with the greatest discriminant (D) value. THAT'S WHY WE REVERSE THE ORDER in computing the comulative density 
        # function (or simply distribution function!)
        # Because we want to start by adding the ones that we're the most sure are b-jets, so the last values of nEntries, 
        # corresponding to greater D values.
        
        
        eff = np.add.accumulate(nEntries[::-1]) / np.sum(nEntries)
        effs.append(eff)

    plt.legend()
    plt.xlabel('$D = \ln [ p_b / (f_c p_c + (1- f_c)p_l ) ]$',fontsize=14)
    plt.ylabel('"Normalized" counts')

    if returnDisc:
        return effs, disc
    else:
        return effs