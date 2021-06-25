import numpy as np

data_dir = "./data"
#Peptide Encoding
def encode(peptides, encoding_scheme, alphabet):
    
    encoded_peptides = []

    for peptide in peptides:

        encoded_peptide = []

        for peptide_letter in peptide:

            for alphabet_letter in alphabet:

                encoded_peptide.append(encoding_scheme[peptide_letter][alphabet_letter])

        encoded_peptides.append(encoded_peptide)

    return np.array(encoded_peptides)
#Error Function
def cumulative_error(peptides, y, lamb, weights):

    error = 0
    
    for i in range(0, len(peptides)):
        
        # get peptide
        peptide = peptides[i]

        # get target prediction value
        y_target = y[i]
        
        # get prediction
        y_pred = np.dot(peptide, weights)
            
        # calculate error
        error += 1.0/2 * (y_pred - y_target)**2
        
    gerror = error + lamb*np.dot(weights, weights)
    error /= len(peptides)
        
    return gerror, error

#Predict value for a peptide list
def predict(peptides, weights):

    pred = []
    
    for i in range(0, len(peptides)):
        
        # get peptide
        peptide = peptides[i]
        
        # get prediction
        y_pred = np.dot(peptide, weights)
        
        pred.append(y_pred)
        
    return pred

#Calculate MSE between two vectors
def cal_mse(vec1, vec2):
    
    mse = 0
    
    for i in range(0, len(vec1)):
        mse += (vec1[i] - vec2[i])**2
        
    mse /= len(vec1)
    
    return( mse)

#Gradient Descent
def gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon):
    
    # do is dE/dO
    do = (y_pred-y_target)
        
    for i in range(0, len(weights)):
        
        de_dw_i = do*peptide[i]+lamb_N*2*weights[i]

        weights[i] -= epsilon*de_dw_i
      
def train(train_data, test_data):
    #Alphabet
    alphabet_file = data_dir + "/Matrices/alphabet"
    alphabet = np.loadtxt(alphabet_file, dtype=str)

    sparse_file = data_dir + "/Matrices/sparse"
    _sparse = np.loadtxt(sparse_file, dtype=float)
    sparse = {} 
            
    blosum_file = data_dir + "/Matrices/BLOSUM50"
    #blosum_file = "https://raw.githubusercontent.com/brunoalvarez89/data/master/algorithms_in_bioinformatics/part_3/blosum50"
    
    _blosum50 = np.loadtxt(blosum_file, dtype=float).reshape((24, -1)).T
    
    blosum50 = {}
    
    for i, letter_1 in enumerate(alphabet):
            
        blosum50[letter_1] = {}
    
        for j, letter_2 in enumerate(alphabet):
                
            blosum50[letter_1][letter_2] = _blosum50[i, j] / 5.0
  # Random seed 
    np.random.seed( 1 )

    # peptides
    peptides = train_data[:, 0]
    peptides = encode(peptides, sparse, alphabet)
    N = len(peptides)

  # target values
    y = np.array(train_data[:, 1], dtype=float)

  #evaluation peptides
    evaluation_peptides = test_data[:, 0]
    evaluation_peptides = encode(evaluation_peptides, blosum50, alphabet)

#evaluation targets
    evaluation_targets = np.array(test_data[:, 1], dtype=float)

# weights
    input_dim  = len(peptides[0])
    output_dim = 1
    w_bound = 0.1
    weights = np.random.uniform(-w_bound, w_bound, size=input_dim)

# training epochs
    epochs = 100

# regularization lambda
#lamb = 1
#lamb = 10
    lamb = 0.01

# regularization lambda per target value
    lamb_N = lamb/N

# learning rate
    epsilon = 0.01

# error  plot
    gerror_plot = []
    mse_plot = []
    train_mse_plot = []
    eval_mse_plot = []
    train_pcc_plot = []
    eval_pcc_plot = []

# for each training epoch
    for e in range(0, epochs):

      # for each peptide
        for i in range(0, N):

          # random index
            ix = np.random.randint(0, N)
          
          # get peptide       
            peptide = peptides[ix]

          # get target prediction value
            y_target = y[ix]
        
          # get initial prediction
            y_pred = np.dot(peptide, weights)

          # gradient descent 
            gradient_descent(y_pred, y_target, peptide, weights, lamb_N, epsilon)

      # compute error
        gerr, mse = cumulative_error(peptides, y, lamb, weights) 
        gerror_plot.append(gerr)
        mse_plot.append(mse)
      
      # predict on training data
        train_pred = predict( peptides, weights )
        train_mse = cal_mse( y, train_pred )
        train_mse_plot.append(train_mse)
        #train_pcc = pearsonr( y, train_pred )
        #train_pcc_plot.append( train_pcc[0] )
          
      # predict on evaluation data
        eval_pred = predict(evaluation_peptides, weights )
        eval_mse = cal_mse(evaluation_targets, eval_pred )
        eval_mse_plot.append(eval_mse)
        #eval_pcc = pearsonr(evaluation_targets, eval_pred)
        #eval_pcc_plot.append( eval_pcc[0] )
      
        # print ("Epoch: ", e, "Gerr:", gerr, train_pcc[0], train_mse, eval_pcc[0], eval_mse)


# our matrices are vectors of dictionaries
    def vector_to_matrix(vector, alphabet):
      
        rows = int(len(vector)/len(alphabet))
      
        matrix = [0] * rows
      
        offset = 0
      
        for i in range(0, rows):
          
            matrix[i] = {}
          
            for j in range(0, 20):
              
                matrix[i][alphabet[j]] = vector[j+offset] 
          
            offset += len(alphabet)

        return matrix 


    def to_psi_blast(matrix):

      # print to user
      
        header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

        print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*header)) 

        letter_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

        for i, row in enumerate(matrix):

            scores = []

            scores.append(str(i+1) + " A")

            for letter in letter_order:

                score = row[letter]

                scores.append(round(score, 4))

            print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*scores)) 


    smm_matrix = vector_to_matrix(weights, alphabet)
    #to_psi_blast(matrix)

    return smm_matrix

def evaluate(evaluation_data, smm_matrix):
  
    def score_peptide(peptide, matrix):
        acum = 0
        for i in range(0, len(peptide)):
            acum += matrix[i][peptide[i]]
        return acum

    # Read evaluation data
    # evaluation = np.loadtxt(evaluation_file, dtype=str).reshape(-1,2)
    evaluation_peptides = evaluation_data[:, 0]
    evaluation_targets = evaluation_data[:, 1].astype(float)

    evaluation_peptides, evaluation_targets

    peptide_length = len(evaluation_peptides[0])

    evaluation_predictions = []
    for i in range(len(evaluation_peptides)):
        score = score_peptide(evaluation_peptides[i], smm_matrix)
        evaluation_predictions.append(score)
        #print (evaluation_peptides[i], score, evaluation_targets[i])
    return(evaluation_predictions)
