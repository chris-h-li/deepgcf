'''
This script generates predictions for a set of pairs given a trained neural network.
The script has been updated by CL to incorporate a third species. Previously the code only incorporated two species, human and pig.
'''
import gzip,random,numpy as np,argparse

# Pytorch
import torch
from torch.autograd import Variable
import torch.utils.data
# CL: import the updated code defining neural network architecture for three species
from shared_deepgcf_three_species import *

# CL added parameters for third species
def predict(net,
            human_test_filename,pig_test_filename, sp3_test_filename, output_filename,
            test_data_size,batch_size,
            num_human_features,num_pig_features, num_sp3_features):

     ### added by CL to activate GPU on Google Colab, can remove if explicit call to use gpu not required -------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ### ---------------------------------------------
    

    # Make predictions and write to output
    with open(human_test_filename,'r') as hf,\
            open(pig_test_filename,'r') as mf,\
            open(sp3_test_filename,'r') as spf,\
            gzip.open(output_filename if output_filename.endswith('.gz') else output_filename+'.gz','wb') as fout:
        next(hf)
        next(mf)
        # CL added
        next(spf)
        for i in range(int(test_data_size/batch_size)+1): # iterate through each batch
            current_batch_size = batch_size if i<int(test_data_size/batch_size) else test_data_size%batch_size

            if current_batch_size==0:
                break

            X = np.zeros((current_batch_size,num_human_features),dtype=float) # to store human data
            Y = np.zeros((current_batch_size,num_pig_features),dtype=float) # to store pig data
            # CL added
            Z = np.zeros((current_batch_size,num_sp3_features),dtype=float) # to store sp3 data

            for j in range(current_batch_size): # iterate through each sample within the batch
                hl = hf.readline().strip().split('\t')
                del hl[0:6]
                for k in range(0, len(hl)):
                    hl[k] = float(hl[k])

                ml = mf.readline().strip().split('\t')
                del ml[0:6]
                for k in range(0, len(ml)):
                    ml[k] = float(ml[k])
                
                # CL added
                spl = spf.readline().strip().split('\t')
                del spl[0:6]
                for k in range(0, len(spl)):
                    spl[k] = float(spl[k])
                
                X[j, :] = hl
                Y[j, :] = ml
                # CL added
                Z[j, :] = spl

            # Convert feature matrices for PyTorch
            X = Variable(torch.from_numpy(X).float())
            Y = Variable(torch.from_numpy(Y).float())
            # CL added
            Z = Variable(torch.from_numpy(Z).float())
            # CL edited to add Z data
            inputs = torch.cat((X,Y, Z),1) # concatenate human and pig and sp3 data
            
             ### added by CL to activate GPU -------------------
            inputs = inputs.to(device) 
            ### ---------------------------------------------
            

            # Make prediction on current batch
            #CL added ".to(device)" to activate gpu
            y_pred = net(inputs).to(device) # put the feature matrix into the provided trained PSNN
            y_pred = y_pred.data
            y_pred_np = torch.Tensor.cpu(y_pred).data.numpy() #changed by CL to convert gpu tensor to numpy

            # Write predicted probabilities of the current batch
            sample_output = [str(np.round(y_pred_np[j],7)) for j in range(current_batch_size)]
            l = '\n'.join(sample_output)+'\n'
            fout.write(l.encode())

def main():
    epilog = '# Example: python src/predict_deepgcf.py -t NN/odd_ensemble/NN_1_*.pt -H data/even_all.h.gz -M \
    data/even_all.m.gz -d 16627449 -o NN/odd_ensemble/even_all_1.gz'
    
    parser = argparse.ArgumentParser(prog='python src/predict_deepgcf_three_species.py',
                                     description='Generate predictions given a trained neural network',
                                     epilog=epilog)
    parser.add_argument('-s', '--seed', help='random seed (default: 1)', type=int, default=1)
    parser.add_argument('-b', '--batch-size', help='batch size (default: 128)', type=int, default=128)

    g1 = parser.add_argument_group('required arguments specifying input and output')
    g1.add_argument('-t', '--trained-classifier-filename', required=True, help='path to a trained classifier (.pt)',
                    type=str)
    g1.add_argument('-H', '--human-feature-filename', required=True, help='path to human feature data file', type=str)
    g1.add_argument('-M', '--pig-feature-filename', required=True, help='path to pig feature data file', type=str)
    # added by CL
    g1.add_argument('-SP', '--sp3-feature-filename', required=True, help='path to sp3 feature data file', type=str)
    g1.add_argument('-d', '--data-size', required=True, help='number of samples', type=int)
    g1.add_argument('-o', '--output-filename', required=True, help='path to output file', type=str)

    g1.add_argument('-hf', '--num-human-features',
                    help='number of human features in input vector (default: 8824)', type=int, default=8824)
    g1.add_argument('-mf', '--num-pig-features',
                    help='number of pig features in input vector (default: 3113)', type=int, default=3113)
    # added by CL
    g1.add_argument('-spf', '--num-sp3-features',
                    help='number of sp3 features in input vector (default: 3113)', type=int, default=3113)

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load previously trained classifier
    net = torch.load(args.trained_classifier_filename)
    net.eval() # make sure it's in evaluation mode

    # Make predictions
    # CL added parameter inputs for third species
    predict(net,
            args.human_feature_filename,args.pig_feature_filename,args.sp3_feature_filename,args.output_filename,
            args.data_size,args.batch_size,
            args.num_human_features,args.num_pig_features, args.num_sp3_features)

if __name__ == "__main__":
    main()
