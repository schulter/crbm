import os
import crbm

def tutorial(path):
    # List of Biopython sequences
    onehot = crbm.load_sample()

    # Obtain a cRBM object
    # Epoch = 1 is not sufficient to train meaningful features
    # however, it speeds up automatic testing.
    model = crbm.CRBM(num_motifs = 10, motif_length = 15, epochs = 1)

    # Fit the model
    model.fit(onehot)

    # Save parameters and hyper-parameters
    model.saveModel(os.path.join(path, 'oct4_model_params.pkl'))

    # Reinstantiate model
    model = crbm.CRBM.loadModel(os.path.join(path, 'oct4_model_params.pkl'))

    # Get a list of numpy matrices representing PFMs
    model.getPFMs()

    # Store the PFMs (by default in 'jaspar' format)
    # in the folder './pfms/'
    crbm.saveMotifs(model, path = os.path.join(path, 'pfms'))

    # Writes all logos in the logos/ directory
    crbm.utils.createSeqLogos(model, path = os.path.join(path, 'logos'))

    # Alternatively, an individual sequence logo can be created:
    # Get first motif
    pfm = model.getPFMs()[0]

    # Create a corresponding sequence logo
    crbm.utils.createSeqLogo(pfm, filename = os.path.join(path, 'logo1.png'), 
            fformat = "png")

    # Per-position motif match probabilities
    # for the first 100 sequences
    matches = model.motifHitProbs(onehot[:100])

    # Plot positional enrichment for all motifs in the given
    # test sequences
    crbm.positionalDensityPlot(model, onehot[:100], 
            filename = os.path.join(path, 'densityplot.png'))

    # Run t-SNE clustering
    tsne = crbm.runTSNE(model, onehot)

    # Visualize the results in a scatter plot
    crbm.tsneScatter({'Oct4': tsne}, 
            filename = os.path.join(path, 'tsnescatter.png'))

    # Visualize the results in the scatter plot
    # by augmenting with the respective motif abundances
    crbm.tsneScatterWithPies(model, onehot, tsne, 
            filename = os.path.join(path, 'tsnescatter_pies.png'))

    # Assemble multiple datasets as follows
    data = {'set1': onehot[:1500], 'set2': onehot[1500:]}

    crbm.violinPlotMotifMatches(model, data, 
            filename = os.path.join(path, 'violinplot.png'))

