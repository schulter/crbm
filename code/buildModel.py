import trainingObserver as observer
from  convRBM import CRBM

def getObserver(model, title):
    for obs in model.observers:
        if title.lower() in obs.name.lower():
            return obs
    return None


def buildModelWithObservers(hyperParams, test_data, veri_data):
    model = CRBM(hyperParams=hyperParams)

    # add the observers for free energy (test and train)
    free_energy_observer = observer.FreeEnergyObserver(model, test_data, "FE-test")
    model.addObserver(free_energy_observer)
    free_energy_train_observer = observer.FreeEnergyObserver(model, veri_data, "FE-training")
    model.addObserver(free_energy_train_observer)

    # add the observers for reconstruction error (test and train)
    #reconstruction_observer = observer.ReconstructionRateObserver(model,
    #                                                             test_data,
    #                                                             "Recon-test")
    #odel.addObserver(reconstruction_observer)
    #econstruction_observer_train = observer.ReconstructionRateObserver(model,
    #                                                                   veri_data,
    #                                                                   "Recon-training")
    #odel.addObserver(reconstruction_observer_train)

    # add the observer of the motifs during training (to look at them afterwards)
    #aram_observer = observer.ParameterObserver(model, None)
    #odel.addObserver(param_observer)

    # add the motif hit scanner
    #motif_hit_observer = observer.MotifHitObserver(model, testingData)
    #odel.addObserver(motif_hit_observer)

    # add IC observers
    icObserver = observer.InformationContentObserver(model)
    model.addObserver(icObserver)

    medianICObserver = observer.MedianICObserver(model)
    model.addObserver(medianICObserver)

    return model

