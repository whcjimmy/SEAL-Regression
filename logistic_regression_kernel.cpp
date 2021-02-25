#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "seal/seal.h"
#include "helper.h"

using namespace std;
using namespace seal;

// Dot Product
double vector_dot_product(vector<double> vec_A, vector<double> vec_B)
{
    if (vec_A.size() != vec_B.size())
    {
        cerr << "Vector size mismatch" << endl;
        exit(1);
    }

    double result = 0;
    for (unsigned int i = 0; i < vec_A.size(); i++)
    {
        result += vec_A[i] * vec_B[i];
    }

    return result;
}


// Linear Transformation (or Matrix * Vector)
vector<double> linear_transformation(vector<vector<double>> input_matrix, vector<double> input_vec)
{

    int rowSize = input_matrix.size();
    int colSize = input_matrix[0].size();

    if (colSize != input_vec.size())
    {
        cerr << "Matrix Vector sizes error" << endl;
        exit(EXIT_FAILURE);
    }

    vector<double> result_vec(rowSize);
    for (int i = 0; i < input_matrix.size(); i++)
    {
        result_vec[i] = vector_dot_product(input_matrix[i], input_vec);
    }

    return result_vec;
}


// String matrix to double matrix converter
vector<vector<double>> stringTodoubleMatrix(vector<vector<string>> matrix)
{
    vector<vector<double>> result(matrix.size(), vector<double>(matrix[0].size()));
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[0].size(); j++)
        {
            result[i][j] = ::atof(matrix[i][j].c_str());
        }
    }

    return result;
}

// Standard Scaler
vector<vector<double>> standard_scaler(vector<vector<double>> input_matrix)
{
    int rowSize = input_matrix.size();
    int colSize = input_matrix[0].size();
    vector<vector<double>> result_matrix(rowSize, vector<double>(colSize));

    // Optimization: Get Means and Standard Devs first then do the scaling
    // first pass: get means and standard devs
    vector<double> means_vec(colSize);
    vector<double> stdev_vec(colSize);
    for (int i = 0; i < colSize; i++)
    {
        vector<double> column(rowSize);
        for (int j = 0; j < rowSize; j++)
        {
            // cout << input_matrix[j][i] << ", ";
            column[j] = input_matrix[j][i];
            // cout << column[j] << ", ";
        }

        means_vec[i] = getMean(column);
        stdev_vec[i] = getStandardDev(column, means_vec[i]);
        // cout << "MEAN at i = " << i << ":\t" << means_vec[i] << endl;
        // cout << "STDV at i = " << i << ":\t" << stdev_vec[i] << endl;
    }

    // second pass: scale
    for (int i = 0; i < rowSize; i++)
    {
        for (int j = 0; j < colSize; j++)
        {
            result_matrix[i][j] = (input_matrix[i][j] - means_vec[j]) / stdev_vec[j];
            // cout << "RESULT at i = " << i << ":\t" << result_matrix[i][j] << endl;
        }
    }

    return result_matrix;
}

// Min Max Scaler
vector<vector<double>> minmax_scaler(vector<vector<double>> input_matrix)
{
    int rowSize = input_matrix.size();
    int colSize = input_matrix[0].size();
    for(int j = 0; j < colSize; j++) {
        double max = -99999;
        double min = 99999;
        for(int i = 0; i < rowSize; i++) {
            if(input_matrix[i][j] > max) {
                max = input_matrix[i][j];
            }
            if(input_matrix[i][j] < min) {
                min = input_matrix[i][j];
            }
        }
        for(int i = 0; i < rowSize; i++) {
            input_matrix[i][j] = (input_matrix[i][j] - min) / (max - min);
        }
    }

    return input_matrix;
}


int main()
{
    int poly_modulus_degree = 16384;
    EncryptionParameters params(scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    cout << "MAX BIT COUNT: " << CoeffModulus::MaxBitCount(poly_modulus_degree) << endl;
    params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 35, 35, 35, 35, 35, 35, 35, 35, 35, 60}));
    SEALContext context(params);

    print_parameters(context);

    cout << "Print the modulus switching chain" << endl;

    // Print the key level parameter info
    auto context_data = context.key_context_data();
    cout << "\tLevel (chain index): " << context_data->chain_index() << endl;
    cout << "\tparms_id: " << context_data->parms_id() << endl;
    cout << "\tcoeff_modulus primes: ";
    cout << hex;
    for (const auto &prime : context_data->parms().coeff_modulus())
    {
        cout << prime.value() << " ";
    }
    cout << dec << endl;
    cout << "\\" << endl;
    cout << " \\-->";

    // Iterate over the remaining levels
    context_data = context.first_context_data();
    while (context_data)
    {
        cout << " Level (chain index): " << context_data->chain_index();
        if (context_data->parms_id() == context.first_parms_id())
        {
            cout << " ...... first_context_data()" << endl;
        }
        else if (context_data->parms_id() == context.last_parms_id())
        {
            cout << " ...... last_context_data()" << endl;
        }
        else
        {
            cout << endl;
        }
        cout << "      parms_id: " << context_data->parms_id() << endl;
        cout << "      coeff_modulus primes: ";
        cout << hex;
        for (const auto &prime : context_data->parms().coeff_modulus())
        {
            cout << prime.value() << " ";
        }
        cout << dec << endl;
        cout << "\\" << endl;
        cout << " \\-->";

        // Step forward in the chain.
        context_data = context_data->next_context_data();
    }

    cout << "End of chain reached\n"
         << endl;

    // Generate keys, encryptor, decryptor and evaluator
    KeyGenerator keygen(context);
    SecretKey sk = keygen.secret_key();
    PublicKey pk;
    keygen.create_public_key(pk);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    Encryptor encryptor(context, pk);
    Evaluator evaluator(context);
    Decryptor decryptor(context, sk);

    // Create CKKS encoder
    CKKSEncoder ckks_encoder(context);

    // Create Scale
    double scale = pow(2.0, 35);

    // Time
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::milliseconds time_diff;

    // read file
    // string filename = "pulsar_stars.csv";
    string filename = "./python/datasets/Heart-Disease-Machine-Learning/data.csv";
    vector<vector<string>> s_matrix = CSVtoMatrix(filename);
    vector<vector<double>> f_matrix = stringTodoubleMatrix(s_matrix);

    // Init features, labels and weights
    // Init features (rows of f_matrix , cols of f_matrix - 1)
    int rows = f_matrix.size();
    // rows = 10;
    cout << "\nNumber of rows  = " << rows << endl;
    int cols = f_matrix[0].size() - 1;
    cout << "\nNumber of cols  = " << cols << endl;

    vector<vector<double>> features(rows, vector<double>(cols));
    // Init labels (rows of f_matrix)
    vector<double> labels(rows);
    // Init weight vector with zeros (cols of features)
    vector<double> beta(rows);

    // Fill the features matrix and labels vector
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            features[i][j] = f_matrix[i][j];
        }
        labels[i] = f_matrix[i][cols];
    }

    // Fill the weights with random numbers (from 1 - 2)
    for (int i = 0; i < rows; i++) {
        beta[i] = RandomFloat(0.0001, 0.0005);
        // beta[i] = RandomFloat(-2, 2) + 0.00000001;
    }

    vector<vector<double>> standard_features = minmax_scaler(features);

    // seperate features into two parts
    int col_A = 4;
    int col_B = cols - col_A;

    vector<vector<double>> kernel(rows, vector<double>(rows));
    vector<vector<double>> kernel_A(rows, vector<double>(rows));
    vector<vector<double>> kernel_B(rows, vector<double>(rows));

    // init to 0
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            kernel_A[i][j] = 0;
            kernel_B[i][j] = 0;
        }
    }
    
    // -------- LINEAR KERNEL --------
    cout << " -------- LINEAR KERNEL -------- " << endl;
    // calculate kernel_A
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            for(int k = 0; k < col_A; k++) { 
                kernel_A[i][j] += 
                    standard_features[i][k] * standard_features[j][k] + 0.00000001;
            }
        }
    }

    // calculate kernel_B
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            for(int k = 0; k < col_B; k++) { 
                kernel_B[i][j] += 
                    standard_features[i][col_A + k] * standard_features[j][col_A + k] + 0.00000001;
            }
        }
    }

    // Combine two kernels together
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            kernel[i][j] = kernel_A[i][j] + kernel_B[i][j];
        }
    }

    /*
    // -------- Polynomial KERNEL --------
    cout << " -------- Polynomial KERNEL -------- " << endl;
    int poly_kernel_deg = 2;
    // calculate kernel_A
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            for(int k = 0; k < col_A; k++) { 
                kernel_A[i][j] += 
                    standard_features[i][k] * standard_features[j][k] + 0.00000001;
            }
            kernel_A[i][j] = kernel_A[i][j] / cols;
        }
    }

    // calculate kernel_B
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            for(int k = 0; k < col_B; k++) { 
                kernel_B[i][j] += 
                    standard_features[i][col_A + k] * standard_features[j][col_A + k] + 0.00000001;
            }
            kernel_B[i][j] = 1 + kernel_B[i][j] / cols;
        }
    }

    // Combine two kernels together
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            kernel[i][j] = kernel_A[i][j] + kernel_B[i][j];
            kernel[i][j] = pow(kernel[i][j], poly_kernel_deg);
        }
    }
    */

    bool is_rbf = false;
    vector<vector<double>> kernel_A_2(rows, vector<double>(rows));
    vector<vector<double>> kernel_B_2(rows, vector<double>(rows));
    /*
    // -------- RBF KERNEL --------
    cout << " -------- RBF KERNEL -------- " << endl;
    is_rbf = true;
    vector<vector<double>> kernel_2(rows, vector<double>(rows));

    // calculate kernel_A
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            for(int k = 0; k < col_A; k++) { 
                kernel_A[i][j] += 
                    pow(standard_features[i][k] - standard_features[j][k], 2);
            }
            kernel_A[i][j] = -1 * kernel_A[i][j] / cols;
            kernel_A_2[i][j] = kernel_A[i][j] / pow(2, 0.5);
        }
    }

    // calculate kernel_B
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            for(int k = 0; k < col_B; k++) { 
                kernel_B[i][j] += 
                    pow(standard_features[i][col_A + k] - standard_features[j][col_A + k], 2);
            }
            kernel_B[i][j] = -1 * kernel_B[i][j] / cols;
            kernel_B_2[i][j] = kernel_B[i][j] / pow(2, 0.5);
        }
    }

    // Combine two kernels together
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            kernel[i][j] = kernel_A[i][j] + kernel_B[i][j];
            kernel_2[i][j] = kernel_A_2[i][j] + kernel_B_2[i][j];
            kernel[i][j] = 1 + kernel[i][j] + pow(kernel_2[i][j], 2);
        }
    }
    */


    // diagonal matrix
    vector<vector<double>> kernel_A_diagonals(rows, vector<double>(rows));
    vector<vector<double>> kernel_B_diagonals(rows, vector<double>(rows));
    vector<vector<double>> kernel_A_2_diagonals(rows, vector<double>(rows));
    vector<vector<double>> kernel_B_2_diagonals(rows, vector<double>(rows));

    for (int i = 0; i < rows; i++) {
        kernel_A_diagonals[i] = get_diagonal(i, kernel_A);
        kernel_B_diagonals[i] = get_diagonal(i, kernel_B);
    }

    if(is_rbf == true) {
        for(int i = 0; i < rows; i++) {
            kernel_A_2_diagonals[i] = get_diagonal(i, kernel_A_2);
            kernel_B_2_diagonals[i] = get_diagonal(i, kernel_B_2);
        }
    }

    double lambda = 0.01;
    double poly_deg = 3;
    // double poly_deg = 7;

    vector<double> coeffs = {0.50091, 0.19832, -0.00018, -0.00447};
    // vector<double> coeffs = {0.50054, 0.19688, -0.00014, -0.00544, 0.000005, 0.000075, -0.00000004, -0.0000003};
    double learning_rate = 0.0001;
    int iter_times = 15;
    
    // Calculate gradient descents in the plaintext domain

    vector<double> beta_1 = beta;
    vector<double> delta_beta(rows, 0.0);
    vector<double> l2_reg(rows, 0.0);
    double b_k, tmp;

    for(int iter = 0; iter < iter_times; iter++) {
        for(int i = 0; i < rows; i++) {
            b_k = vector_dot_product(beta_1, kernel[i]);
            tmp = 0.0;

            for(int j = 0; j <= poly_deg; j++) {
                tmp = coeffs[j] * pow(-1 * labels[i], j + 1) / rows;
                tmp = tmp * pow(b_k, j);
                for(int k = 0; k < rows; k++) {
                    delta_beta[k] += tmp * kernel[i][k];
                }
            }
        }

        l2_reg = linear_transformation(kernel, beta_1);
        tmp = 2 * lambda / rows;
        for(int i = 0; i < rows; i++) {
            beta_1[i] = beta_1[i] - learning_rate * (l2_reg[i] * tmp + delta_beta[i]);
        }

        // output results
        cout << "iter " << iter << endl;
        for(int i = 0; i < rows; i++) {
            cout << beta_1[i] << " ";
        }
        cout << endl;

    }


    // Calculate gradient descents in the encrypted domain

    // --------------- ENCODING ----------------
    time_start = chrono::high_resolution_clock::now();

    cout << "ENCODING......\n";
    vector<Plaintext> kernel_A_plain(rows), kernel_B_plain(rows);
    vector<Plaintext> kernel_A_D_plain(rows), kernel_B_D_plain(rows);
    vector<Plaintext> kernel_A_2_plain(rows), kernel_B_2_plain(rows);
    vector<Plaintext> kernel_A_2_D_plain(rows), kernel_B_2_D_plain(rows);

    for (int i = 0; i < rows; i++) {
        ckks_encoder.encode(kernel_A[i], scale, kernel_A_plain[i]);
        ckks_encoder.encode(kernel_B[i], scale, kernel_B_plain[i]);
        ckks_encoder.encode(kernel_A_diagonals[i], scale, kernel_A_D_plain[i]);
        ckks_encoder.encode(kernel_B_diagonals[i], scale, kernel_B_D_plain[i]);
    }

    if(is_rbf == true) {
        for (int i = 0; i < rows; i++) {
            ckks_encoder.encode(kernel_A_2[i], scale, kernel_A_2_plain[i]);
            ckks_encoder.encode(kernel_B_2[i], scale, kernel_B_2_plain[i]);
            ckks_encoder.encode(kernel_A_2_diagonals[i], scale, kernel_A_2_D_plain[i]);
            ckks_encoder.encode(kernel_B_2_diagonals[i], scale, kernel_B_2_D_plain[i]);
        }
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << rows << " total encoding time :\t" << time_diff.count() << " milliseconds" << endl;

    // --------------- ENCRYPTNG ------------------
    time_start = chrono::high_resolution_clock::now();

    cout << "ENCRYPTING......\n";
    vector<Ciphertext> kernel_A_cipher(rows), kernel_B_cipher(rows);
    vector<Ciphertext> kernel_A_D_cipher(rows), kernel_B_D_cipher(rows);
    vector<Ciphertext> kernel_A_2_cipher(rows), kernel_B_2_cipher(rows);
    vector<Ciphertext> kernel_A_2_D_cipher(rows), kernel_B_2_D_cipher(rows);

    for (int i = 0; i < rows; i++) {
        encryptor.encrypt(kernel_A_plain[i], kernel_A_cipher[i]);
        encryptor.encrypt(kernel_B_plain[i], kernel_B_cipher[i]);
        encryptor.encrypt(kernel_A_D_plain[i], kernel_A_D_cipher[i]);
        encryptor.encrypt(kernel_B_D_plain[i], kernel_B_D_cipher[i]);
    }

    if(is_rbf == true) {
        for (int i = 0; i < rows; i++) {
            encryptor.encrypt(kernel_A_2_plain[i], kernel_A_2_cipher[i]);
            encryptor.encrypt(kernel_B_2_plain[i], kernel_B_2_cipher[i]);
            encryptor.encrypt(kernel_A_2_D_plain[i], kernel_A_2_D_cipher[i]);
            encryptor.encrypt(kernel_B_2_D_plain[i], kernel_B_2_D_cipher[i]);
        }
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << rows << " total encryption time :\t" << time_diff.count() << " milliseconds" << endl;
   
    // --------------- CALCULATTNG ------------------
    time_start = chrono::high_resolution_clock::now();

    double one = 1;
    Plaintext one_plain;
    Ciphertext one_cipher;
    ckks_encoder.encode(one, scale, one_plain);
    encryptor.encrypt(one_plain, one_cipher);

    cout << "CALCULATING......\n";
    vector<Ciphertext> kernel_cipher(rows);  // x
    vector<Ciphertext> kernel_diagonals_cipher(rows);  // x diagonal

    // LINEAR KERNEL
    for(int i = 0; i < rows; i++) {
        evaluator.add(kernel_A_cipher[i], kernel_B_cipher[i], kernel_cipher[i]);
        evaluator.add(kernel_A_D_cipher[i], kernel_B_D_cipher[i], kernel_diagonals_cipher[i]);
    }

    /*
    // POLYNOMIAL KERNEL
    for(int i = 0; i < rows; i++) {
        evaluator.add(kernel_A_cipher[i], kernel_B_cipher[i], kernel_cipher[i]);
        evaluator.add(kernel_A_D_cipher[i], kernel_B_D_cipher[i], kernel_diagonals_cipher[i]);
    }
    vector<Ciphertext> kernel_powers_cipher(poly_kernel_deg);
    vector<Ciphertext> kernel_D_powers_cipher(poly_kernel_deg);
    for(int i = 0; i < rows; i++) {
        // original
        compute_all_powers(kernel_cipher[i], poly_kernel_deg, evaluator, relin_keys, kernel_powers_cipher);
        kernel_cipher[i] = kernel_powers_cipher[poly_kernel_deg];

        // diagonal
        compute_all_powers(kernel_diagonals_cipher[i], poly_kernel_deg, evaluator, relin_keys, kernel_D_powers_cipher);
        kernel_diagonals_cipher[i] = kernel_D_powers_cipher[poly_kernel_deg];

        // Test
        // Plaintext kernel_powers_plaintext;
        // vector<double> kernel_powers_decode;
        // decryptor.decrypt(kernel_powers_cipher[poly_kernel_deg], kernel_powers_plaintext);
        // ckks_encoder.decode(kernel_powers_plaintext, kernel_powers_decode);
    }
    */
    
    /*
    // RBF KERNEL
    if(is_rbf == true) {
        vector<Ciphertext> kernel_2_cipher(rows);
        vector<Ciphertext> kernel_2_D_cipher(rows);
        for(int i = 0; i < rows; i++) {
            // original
            evaluator.add(kernel_A_2_cipher[i], kernel_B_2_cipher[i], kernel_2_cipher[i]);
            evaluator.multiply_inplace(kernel_2_cipher[i], kernel_2_cipher[i]);
            evaluator.rescale_to_next_inplace(kernel_2_cipher[i]);
            evaluator.relinearize_inplace(kernel_2_cipher[i], relin_keys);

            evaluator.add(kernel_A_cipher[i], kernel_B_cipher[i], kernel_cipher[i]);

            evaluator.mod_switch_to_inplace(one_plain, kernel_2_cipher[i].parms_id());
            evaluator.mod_switch_to_inplace(kernel_cipher[i], kernel_2_cipher[i].parms_id());

            evaluator.add_plain_inplace(kernel_cipher[i], one_plain);
            kernel_2_cipher[i].scale() = pow(2, (int)log2(kernel_cipher[i].scale()));
            evaluator.add_inplace(kernel_cipher[i], kernel_2_cipher[i]);

            // diagonal
            evaluator.add(kernel_A_2_cipher[i], kernel_B_2_cipher[i], kernel_2_D_cipher[i]);
            evaluator.multiply_inplace(kernel_2_D_cipher[i], kernel_2_D_cipher[i]);
            evaluator.rescale_to_next_inplace(kernel_2_D_cipher[i]);
            evaluator.relinearize_inplace(kernel_2_D_cipher[i], relin_keys);

            evaluator.add(kernel_A_cipher[i], kernel_B_cipher[i], kernel_diagonals_cipher[i]);

            evaluator.mod_switch_to_inplace(one_plain, kernel_2_D_cipher[i].parms_id());
            evaluator.mod_switch_to_inplace(kernel_diagonals_cipher[i], kernel_2_D_cipher[i].parms_id());

            evaluator.add_plain_inplace(kernel_diagonals_cipher[i], one_plain);
            kernel_2_D_cipher[i].scale() = pow(2, (int)log2(kernel_diagonals_cipher[i].scale()));
            evaluator.add_inplace(kernel_diagonals_cipher[i], kernel_2_D_cipher[i]);
        }
    }
    */

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << rows << " total kernel computation time :\t" << time_diff.count() << " milliseconds" << endl;


    // --------------- CALCULATTNG ------------------
    time_start = chrono::high_resolution_clock::now();

    cout << "CALCULATING......\n";

    double alpha;
    Plaintext alpha_plain, l2_reg_alpha_plain, beta_plain;
    Ciphertext x_cipher, beta_cipher, beta_kernel_cipher, delta_beta_all_cipher, l2_reg_cipher;
    vector<Ciphertext> delta_beta_cipher(rows);
    vector<Ciphertext> bk_powers_cipher(poly_deg);
    // used when decoding
    Plaintext delta_beta_plain;
    vector<double> delta_beta_decode(cols);

    time_start = chrono::high_resolution_clock::now();

    double l2_reg_alpha = 2.0 * lambda / rows;
    ckks_encoder.encode(l2_reg_alpha, scale, l2_reg_alpha_plain);
    

    for(int iter = 0; iter < iter_times; iter++) {
        cout << "iter " << iter << endl;
        ckks_encoder.encode(beta, scale, beta_plain);
        for(int i = 0; i < rows; i++) {
            x_cipher = kernel_cipher[i];
            evaluator.mod_switch_to_inplace(beta_plain, x_cipher.parms_id());

            evaluator.multiply_plain(x_cipher, beta_plain, beta_kernel_cipher);
            evaluator.rescale_to_next_inplace(beta_kernel_cipher);

            compute_all_powers(beta_kernel_cipher, poly_deg, evaluator, relin_keys, bk_powers_cipher);
            evaluator.mod_switch_to_inplace(one_cipher, x_cipher.parms_id());
            bk_powers_cipher[0] = one_cipher;

            for(int j = 0; j <= poly_deg; j++) {
                alpha = coeffs[j] * pow(-1 * labels[i], j + 1) / rows;
                ckks_encoder.encode(alpha, scale, alpha_plain);

                evaluator.mod_switch_to_inplace(alpha_plain, bk_powers_cipher[j].parms_id());
                evaluator.multiply_plain_inplace(bk_powers_cipher[j], alpha_plain);
                evaluator.rescale_to_next_inplace(bk_powers_cipher[j]);

                evaluator.mod_switch_to_inplace(x_cipher, bk_powers_cipher[j].parms_id());
                evaluator.multiply_inplace(bk_powers_cipher[j], x_cipher);
            }

            int last_id = bk_powers_cipher.size() - 1;
            parms_id_type last_parms_id = bk_powers_cipher[last_id].parms_id();
            double last_scale = pow(2, (int)log2(bk_powers_cipher[last_id].scale()));

            for(int j = 0; j <= poly_deg; j++) {
                evaluator.mod_switch_to_inplace(bk_powers_cipher[j], last_parms_id);
                bk_powers_cipher[j].scale() = last_scale;
            }

            evaluator.add_many(bk_powers_cipher, delta_beta_cipher[i]);
        }

        evaluator.add_many(delta_beta_cipher, delta_beta_all_cipher);

        // L2 term
        encryptor.encrypt(beta_plain, beta_cipher);
        l2_reg_cipher = Linear_Transform_Cipher(beta_cipher, kernel_diagonals_cipher, gal_keys, params);
        evaluator.rescale_to_next_inplace(l2_reg_cipher);
        evaluator.mod_switch_to_inplace(l2_reg_alpha_plain, l2_reg_cipher.parms_id());
        evaluator.multiply_plain_inplace(l2_reg_cipher, l2_reg_alpha_plain);

        evaluator.mod_switch_to_inplace(l2_reg_cipher, delta_beta_all_cipher.parms_id());
        l2_reg_cipher.scale() = pow(2, (int)log2(delta_beta_all_cipher.scale()));

        evaluator.add_inplace(delta_beta_all_cipher, l2_reg_cipher);


        // Test
        decryptor.decrypt(delta_beta_all_cipher, delta_beta_plain);
        ckks_encoder.decode(delta_beta_plain, delta_beta_decode);

        for(int i = 0; i < rows; i++) {
            beta[i] = beta[i] - learning_rate * delta_beta_decode[i];
            cout << beta[i] << " ";
        }
        cout << endl;
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << rows << " total execution time :\t" << time_diff.count() << " milliseconds" << endl;

    // acuracy
    double acc_1 = 0.0, acc_2 = 0.0;
    for(int i = 0; i < rows; i++) {
        double tmp_1, tmp_2;
        tmp_1 = vector_dot_product(beta_1, kernel[i]);
        tmp_2 = vector_dot_product(beta, kernel[i]);
        if(tmp_1 >= 0) {
            tmp_1 = 1;
        } else {
            tmp_1 = -1;
        }

        if(tmp_2 >= 0) {
            tmp_2 = 1;
        } else {
            tmp_2 = -1;
        }

        if(tmp_1 == labels[i]) acc_1 += 1;
        if(tmp_2 == labels[i]) acc_2 += 1;
    }
    cout << "acc 1 " << acc_1 / rows << endl;
    cout << "acc 2 " << acc_2 / rows << endl;

    return 0;
}
