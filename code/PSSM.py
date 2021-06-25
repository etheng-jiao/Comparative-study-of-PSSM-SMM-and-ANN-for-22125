import numpy as np
import math
import copy

data_dir = "./data/"

def train(peptides, beta = 50, sequence_weighting = True ):
  
  #Train the PSSM algorithm
  # Alphabet
    alphabet_file = data_dir + "Matrices/alphabet"
    alphabet = np.loadtxt(alphabet_file, dtype=str)

  # Background Frequencies
    bg_file = data_dir + "Matrices/bg.freq.fmt"
    _bg = np.loadtxt(bg_file, dtype=float)

    bg = {}
    for i in range(0, len(alphabet)):
      bg[alphabet[i]] = _bg[i]

  # Blosum62 Matrix
    blosum62_file = data_dir + "Matrices/blosum62.freq_rownorm"
    _blosum62 = np.loadtxt(blosum62_file, dtype=float).T 

    blosum62 = {}

    for i, letter_1 in enumerate(alphabet):
      
        blosum62[letter_1] = {}

        for j, letter_2 in enumerate(alphabet):
          
            blosum62[letter_1][letter_2] = _blosum62[i, j]

  

    boundary=1-math.log(500)/math.log(50000)

    if len(peptides) == 1:
        peptide_length = len(peptides[0])
        peptides = [peptides]
    else:
        peptide_length = len(peptides_f[0][0])

    for i in range(0, len(peptides)):
        if len(peptides_f[i][0]) != peptide_length:
            print("Error, peptides differ in length!")
        if float(peptides[i][1])>=boundary:
            dataset.append(peptides_f[i])
    for i in range(len(dataset)):
        peptides = []
        peptides.append(dataset[i][0])


  # Initialize Matrix
    def initialize_matrix(peptide_length, alphabet):

        init_matrix = [0]*peptide_length

        for i in range(0, peptide_length):

            row = {}

            for letter in alphabet: 
                row[letter] = 0.0
          
            init_matrix[i] = row
          
        return init_matrix

  # Amino Acid Count Matrix (c)
    c_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(0, peptide_length):
          
        for peptide in peptides:
          
            c_matrix[position][peptide[position]] += 1

  # Sequence weighting 

    weights = {}

    for peptide in peptides:

      # apply sequence weighting
        if sequence_weighting:
      
            w = 0.0
            neff = 0.0
          
            for position in range(0, peptide_length):

                r = 0

                for letter in alphabet:        

                    if c_matrix[position][letter] != 0:
                      
                        r += 1

                s = c_matrix[position][peptide[position]]

                w += 1.0/(r * s)

                neff += r
                  
            neff = neff / peptide_length
    
      # do not apply sequence weighting
        else:
          
            w = 1  
          
            neff = len(peptides)  
        

        weights[peptide] = w

  # Observed Frequencies Matrix (f)
    f_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(0, peptide_length):
      
        n = 0;
      
        for peptide in peptides:
        
            f_matrix[position][peptide[position]] += weights[peptide]
        
            n += weights[peptide]
            
        for letter in alphabet: 
            
            f_matrix[position][letter] = f_matrix[position][letter]/n

  # Pseudo Frequencies Matrix (g)
    g_matrix = initialize_matrix(peptide_length, alphabet)

    for position in range(0, peptide_length):

        for letter_1 in alphabet:
            for letter_2 in alphabet:
            
              g_matrix[position][letter_1] += f_matrix[position][letter_2]*blosum62[letter_1][letter_2]

  # Combined Frequencies Matrix (p)

    p_matrix = initialize_matrix(peptide_length, alphabet)

    alpha = neff - 1

    for position in range(0, peptide_length):

        for a in alphabet:
            p_matrix[position][a] = (alpha * f_matrix[position][a] + beta * g_matrix[position][a]) / (alpha + beta)

    # Log Odds Weight Matrix (w)
    w_matrix = initialize_matrix(peptide_length, alphabet)
    for position in range(0, peptide_length):
        
        for letter in alphabet:
            if p_matrix[position][letter] > 0:
                w_matrix[position][letter] = 2 * math.log(p_matrix[position][letter]/bg[letter])/math.log(2)
            else:
                w_matrix[position][letter] = -999.9
    return w_matrix

# """
#   # Write Matrix to PSI-BLAST format
#     def to_psi_blast(matrix):

#         header = ["", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

#         print ('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*header)) 

#         letter_order = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

#         for i, row in enumerate(matrix):

#             scores = []

#             scores.append(str(i+1) + " A")

#             for letter in letter_order:

#                 score = row[letter]

#                 scores.append(round(score, 4))

#             print('{:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(*scores)) 

#   # Define PSSM matrix 
#   pssm_matrix = to_psi_blast(w_matrix) 
# """


#####################################
# Prediction algorithm

def evaluate(evaluation, w_matrix):
  
  def score_peptide(peptide, matrix):
    acum = 0
    for i in range(0, len(peptide)):
        acum += matrix[i][peptide[i]]
    return acum

  # Read evaluation data
  # evaluation = np.loadtxt(evaluation_file, dtype=str).reshape(-1,2)
  evaluation_peptides = evaluation[:, 0]
  evaluation_targets = evaluation[:, 1].astype(float)

  evaluation_peptides, evaluation_targets

  peptide_length = len(evaluation_peptides[0])

  evaluation_predictions = []
  for i in range(len(evaluation_peptides)):
    score = score_peptide(evaluation_peptides[i], w_matrix)
    evaluation_predictions.append(score)
    #print (evaluation_peptides[i], score, evaluation_targets[i])
  return(evaluation_predictions)
