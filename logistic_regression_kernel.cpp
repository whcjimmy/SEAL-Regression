#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "seal/seal.h"
#include "helper.h"
#include "util.cpp"

using namespace std;
using namespace seal;


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

    // Read File
    // string filename = "pulsar_stars.csv";
    // string filename = "./python/datasets/Heart-Disease-Machine-Learning/data.csv";
    string filename = "./python/datasets/breast_cancer/data.csv";
    vector<vector<string>> s_matrix = CSVtoMatrix(filename);
    vector<vector<double>> f_matrix = stringTodoubleMatrix(s_matrix);
    // random_shuffle(f_matrix.begin(), f_matrix.end());

    // Init features, labels and weights
    int total_rows = f_matrix.size();
    cout << "\nNumber of rows  = " << total_rows << endl;
    int total_cols = f_matrix[0].size() - 1;
    cout << "\nNumber of cols  = " << total_cols << endl;

    vector<vector<double>> features(total_rows, vector<double>(total_cols));

    int training_rows = 364;
    int testing_rows = total_rows - training_rows;
    vector<vector<double>> training_x(training_rows, vector<double>(total_cols));
    vector<vector<double>> testing_x(testing_rows, vector<double>(total_cols));
    vector<double> training_y(training_rows);
    vector<double> testing_y(testing_rows);

    // Fill the features matrix and labels vector
    for (int i = 0; i < total_rows; i++) {
        for (int j = 0; j < total_cols; j++) {
            if(i < training_rows) {
                training_x[i][j] = f_matrix[i][j];
            } else {
                testing_x[i - training_rows][j] = f_matrix[i][j];
            }
        }
        if(i < training_rows) {
            training_y[i] = f_matrix[i][total_cols];
        } else {
            testing_y[i - training_rows] = f_matrix[i][total_cols];
        }
    }

    // Init weights
    // Init weight vector with zeros (cols of features)
    vector<double> beta(training_rows);
    for (int i = 0; i < training_rows; i++) {
        beta[i] = 0;
        // weights[i] = RandomFloat(-0.1, 0.1);
    }

    // Polynomial Simulated Sigmoid Function
    double poly_deg = 3;
    // double poly_deg = 7;

    vector<double> coeffs = {0.50081, 0.08937, -0.00001, -0.00297};
    // vector<double> coeffs = {0.50054, 0.19688, -0.00014, -0.00544, 0.000005, 0.000075, -0.00000004, -0.0000003};


    // Parameters Settings
    int col_A = 15;
    int col_B = total_cols - col_A;

    double learning_rate = 0.01;
    int iter_times = 30;
    double gamma  = 0.1;
    double lambda = 0.01;
    int poly_kernel_deg = 3;

    // seperate features into two parts
    vector<vector<double>> testing_kernel(testing_rows, vector<double>(training_rows));
    vector<vector<double>> kernel_A(training_rows, vector<double>(training_rows));
    vector<vector<double>> kernel_B(training_rows, vector<double>(training_rows));
    
    // init to 0
    for(int i = 0; i < training_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            kernel_A[i][j] = 0;
            kernel_B[i][j] = 0;
        }
    }

    // -------- LINEAR KERNEL --------
    cout << " -------- LINEAR KERNEL -------- " << endl;
    // calculate kernel_A
    for(int i = 0; i < training_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < col_A; k++) { 
                kernel_A[i][j] += training_x[i][k] * training_x[j][k];
            }
        }
    }

    // calculate kernel_B
    for(int i = 0; i < training_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < col_B; k++) { 
                kernel_B[i][j] += training_x[i][col_A + k] * training_x[j][col_A + k];
            }
        }
    }
    // Testing Kernel
    for(int i = 0; i < testing_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            testing_kernel[i][j] = vector_dot_product(testing_x[i], training_x[j]);
        }
    }

    /*
    // -------- Polynomial KERNEL --------
    cout << " -------- Polynomial KERNEL -------- " << endl;
    // calculate kernel_A
    for(int i = 0; i < training_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < col_A; k++) { 
                kernel_A[i][j] += training_x[i][k] * training_x[j][k];
            }
            kernel_A[i][j] = kernel_A[i][j] * gamma;
        }
    }

    // calculate kernel_B
    for(int i = 0; i < training_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < col_B; k++) { 
                kernel_B[i][j] += training_x[i][col_A + k] * training_x[j][col_A + k];
            }
            kernel_B[i][j] = kernel_B[i][j] * gamma;
        }
    }
    // Testing Kernel
    for(int i = 0; i < testing_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < total_cols; k++) { 
                testing_kernel[i][j] += testing_x[i][k] * training_x[j][k];
            }
            testing_kernel[i][j] = testing_kernel[i][j] * gamma;
            testing_kernel[i][j] = 1 + testing_kernel[i][j];
            testing_kernel[i][j] = pow(testing_kernel[i][j], poly_kernel_deg);
        }
    }
    */

    bool is_rbf = false;
    vector<vector<double>> kernel_A_2(training_rows, vector<double>(training_rows));
    vector<vector<double>> kernel_B_2(training_rows, vector<double>(training_rows));
    /*
    // -------- RBF KERNEL --------
    cout << " -------- RBF KERNEL -------- " << endl;
    is_rbf = true;
    gamma = 0.5;
    vector<vector<double>> kernel_2(training_rows, vector<double>(training_rows));

    // calculate kernel_A
    for(int i = 0; i < training_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < col_A; k++) { 
                kernel_A[i][j] += 
                    pow(standard_features[i][k] - standard_features[j][k], 2);
            }
            kernel_A[i][j] = -1 * kernel_A[i][j] * gamma;
            kernel_A_2[i][j] = kernel_A[i][j] / pow(2, 0.5);
        }
    }

    // calculate kernel_B
    for(int i = 0; i < training_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < col_B; k++) { 
                kernel_B[i][j] += 
                    pow(standard_features[i][col_A + k] - standard_features[j][col_A + k], 2);
            }
            kernel_B[i][j] = -1 * kernel_B[i][j] * gamma;
            kernel_B_2[i][j] = kernel_B[i][j] / pow(2, 0.5);
        }
    }
    // Testing Kernel
    for(int i = 0; i < testing_rows; i++) {
        for (int j = 0; j < training_rows; j++) { 
            for(int k = 0; k < total_cols; k++) { 
                testing_kernel[i][j] += pow(testing_x[i][k] - training_x[j][k], 2);
            }
            testing_kernel[i][j] = -1 * testing_kernel[i][j] * gamma;
            testing_kernel[i][j] = 1 + testing_kernel[i][j] + pow(testing_kernel[i][j], 2) / 2;
        }
    }
    */

    // diagonal matrix
    vector<vector<double>> kernel_A_diagonals(training_rows, vector<double>(training_rows));
    vector<vector<double>> kernel_B_diagonals(training_rows, vector<double>(training_rows));
    vector<vector<double>> kernel_A_2_diagonals(training_rows, vector<double>(training_rows));
    vector<vector<double>> kernel_B_2_diagonals(training_rows, vector<double>(training_rows));

    for (int i = 0; i < training_rows; i++) {
        kernel_A_diagonals[i] = get_diagonal(i, kernel_A);
        kernel_B_diagonals[i] = get_diagonal(i, kernel_B);
    }

    if(is_rbf == true) {
        for(int i = 0; i < training_rows; i++) {
            kernel_A_2_diagonals[i] = get_diagonal(i, kernel_A_2);
            kernel_B_2_diagonals[i] = get_diagonal(i, kernel_B_2);
        }
    }
    
    // Calculate gradient descents in the encrypted domain

    // --------------- ENCODING ----------------
    time_start = chrono::high_resolution_clock::now();

    cout << "ENCODING......\n";
    vector<Plaintext> kernel_A_plain(training_rows), kernel_B_plain(training_rows);
    vector<Plaintext> kernel_A_D_plain(training_rows), kernel_B_D_plain(training_rows);
    vector<Plaintext> kernel_A_2_plain(training_rows), kernel_B_2_plain(training_rows);
    vector<Plaintext> kernel_A_2_D_plain(training_rows), kernel_B_2_D_plain(training_rows);

    for (int i = 0; i < training_rows; i++) {
        ckks_encoder.encode(kernel_A[i], scale, kernel_A_plain[i]);
        ckks_encoder.encode(kernel_B[i], scale, kernel_B_plain[i]);
        ckks_encoder.encode(kernel_A_diagonals[i], scale, kernel_A_D_plain[i]);
        ckks_encoder.encode(kernel_B_diagonals[i], scale, kernel_B_D_plain[i]);
    }

    if(is_rbf == true) {
        for (int i = 0; i < training_rows; i++) {
            ckks_encoder.encode(kernel_A_2[i], scale, kernel_A_2_plain[i]);
            ckks_encoder.encode(kernel_B_2[i], scale, kernel_B_2_plain[i]);
            ckks_encoder.encode(kernel_A_2_diagonals[i], scale, kernel_A_2_D_plain[i]);
            ckks_encoder.encode(kernel_B_2_diagonals[i], scale, kernel_B_2_D_plain[i]);
        }
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << training_rows << " total encoding time :\t" << time_diff.count() << " milliseconds" << endl;

    // --------------- ENCRYPTNG ------------------
    time_start = chrono::high_resolution_clock::now();

    cout << "ENCRYPTING......\n";
    vector<Ciphertext> kernel_A_cipher(training_rows), kernel_B_cipher(training_rows);
    vector<Ciphertext> kernel_A_D_cipher(training_rows), kernel_B_D_cipher(training_rows);
    vector<Ciphertext> kernel_A_2_cipher(training_rows), kernel_B_2_cipher(training_rows);
    vector<Ciphertext> kernel_A_2_D_cipher(training_rows), kernel_B_2_D_cipher(training_rows);

    for (int i = 0; i < training_rows; i++) {
        encryptor.encrypt(kernel_A_plain[i], kernel_A_cipher[i]);
        encryptor.encrypt(kernel_B_plain[i], kernel_B_cipher[i]);
        encryptor.encrypt(kernel_A_D_plain[i], kernel_A_D_cipher[i]);
        encryptor.encrypt(kernel_B_D_plain[i], kernel_B_D_cipher[i]);
    }

    if(is_rbf == true) {
        for (int i = 0; i < training_rows; i++) {
            encryptor.encrypt(kernel_A_2_plain[i], kernel_A_2_cipher[i]);
            encryptor.encrypt(kernel_B_2_plain[i], kernel_B_2_cipher[i]);
            encryptor.encrypt(kernel_A_2_D_plain[i], kernel_A_2_D_cipher[i]);
            encryptor.encrypt(kernel_B_2_D_plain[i], kernel_B_2_D_cipher[i]);
        }
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << training_rows << " total encryption time :\t" << time_diff.count() << " milliseconds" << endl;
   
    // --------------- CALCULATTNG ------------------
    time_start = chrono::high_resolution_clock::now();

    double one = 1;
    Plaintext one_plain;
    Ciphertext one_cipher;
    ckks_encoder.encode(one, scale, one_plain);
    encryptor.encrypt(one_plain, one_cipher);

    cout << "CALCULATING......\n";
    vector<Ciphertext> kernel_cipher(training_rows);  // x
    vector<Ciphertext> kernel_diagonals_cipher(training_rows);  // x diagonal

    // LINEAR KERNEL
    for(int i = 0; i < training_rows; i++) {
        evaluator.add(kernel_A_cipher[i], kernel_B_cipher[i], kernel_cipher[i]);
        evaluator.add(kernel_A_D_cipher[i], kernel_B_D_cipher[i], kernel_diagonals_cipher[i]);
    }
    
    /*
    // POLYNOMIAL KERNEL
    for(int i = 0; i < training_rows; i++) {
        evaluator.add(kernel_A_cipher[i], kernel_B_cipher[i], kernel_cipher[i]);
        evaluator.add(kernel_A_D_cipher[i], kernel_B_D_cipher[i], kernel_diagonals_cipher[i]);
    }
    vector<Ciphertext> kernel_powers_cipher(poly_kernel_deg);
    vector<Ciphertext> kernel_D_powers_cipher(poly_kernel_deg);
    for(int i = 0; i < training_rows; i++) {
        // original
        compute_all_powers(kernel_cipher[i], poly_kernel_deg, evaluator, relin_keys, kernel_powers_cipher);
        kernel_cipher[i] = kernel_powers_cipher[poly_kernel_deg];

        // diagonal
        compute_all_powers(kernel_diagonals_cipher[i], poly_kernel_deg, evaluator, relin_keys, kernel_D_powers_cipher);
        kernel_diagonals_cipher[i] = kernel_D_powers_cipher[poly_kernel_deg];
    }
    */
    
    /*
    // RBF KERNEL
    if(is_rbf == true) {
        vector<Ciphertext> kernel_2_cipher(training_rows);
        vector<Ciphertext> kernel_2_D_cipher(training_rows);
        for(int i = 0; i < training_rows; i++) {
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
    cout << training_rows << " total kernel computation time :\t" << time_diff.count() << " milliseconds" << endl;


    // --------------- CALCULATTNG ------------------
    time_start = chrono::high_resolution_clock::now();

    cout << "CALCULATING......\n";

    double alpha;
    Plaintext alpha_plain, l2_reg_alpha_plain, beta_plain;
    Ciphertext x_cipher, beta_cipher_1, beta_cipher_2, beta_kernel_cipher, delta_beta_all_cipher, l2_reg_cipher;
    vector<Ciphertext> delta_beta_cipher(training_rows);
    vector<Ciphertext> bk_powers_cipher(poly_deg);
    // used when decoding
    Plaintext delta_beta_plain;
    vector<double> delta_beta_decode(training_rows);

    time_start = chrono::high_resolution_clock::now();

    double l2_reg_alpha = 2.0 * lambda / training_rows;
    ckks_encoder.encode(l2_reg_alpha, scale, l2_reg_alpha_plain);
    

    for(int iter = 0; iter < iter_times; iter++) {
        cout << "iter " << iter << endl;
        ckks_encoder.encode(beta, scale, beta_plain);

        for(int i = 0; i < training_rows; i++) {
            x_cipher = kernel_cipher[i];
            encryptor.encrypt(beta_plain, beta_cipher_1);
            evaluator.mod_switch_to_inplace(beta_cipher_1, x_cipher.parms_id());

            beta_kernel_cipher = cipher_dot_product(x_cipher, beta_cipher_1, training_rows, relin_keys, gal_keys, evaluator);

            compute_all_powers(beta_kernel_cipher, poly_deg, evaluator, relin_keys, bk_powers_cipher);
            evaluator.mod_switch_to_inplace(one_cipher, x_cipher.parms_id());
            bk_powers_cipher[0] = one_cipher;

            int last_id = bk_powers_cipher.size() - 1;
            parms_id_type last_parms_id = bk_powers_cipher[last_id].parms_id();
            double last_scale = pow(2, (int)log2(bk_powers_cipher[last_id].scale()));

            for(int j = 0; j <= poly_deg; j++) {
                evaluator.mod_switch_to_inplace(bk_powers_cipher[j], last_parms_id);
                bk_powers_cipher[j].scale() = last_scale;
            }

            for(int j = 0; j <= poly_deg; j++) {
                alpha = coeffs[j] * pow(-1 * training_y[i], j + 1) / training_rows;
                ckks_encoder.encode(alpha, scale, alpha_plain);

                evaluator.mod_switch_to_inplace(alpha_plain, bk_powers_cipher[j].parms_id());
                evaluator.multiply_plain_inplace(bk_powers_cipher[j], alpha_plain);
                evaluator.rescale_to_next_inplace(bk_powers_cipher[j]);
            }

            evaluator.add_many(bk_powers_cipher, delta_beta_cipher[i]);

            evaluator.mod_switch_to_inplace(x_cipher, delta_beta_cipher[i].parms_id());
            evaluator.multiply_inplace(delta_beta_cipher[i], x_cipher);

        }

        evaluator.add_many(delta_beta_cipher, delta_beta_all_cipher);

        // L2 term
        encryptor.encrypt(beta_plain, beta_cipher_2);
        evaluator.mod_switch_to_inplace(beta_cipher_2, kernel_diagonals_cipher[0].parms_id());

        l2_reg_cipher = Linear_Transform_Cipher(beta_cipher_2, kernel_diagonals_cipher, gal_keys, params);
        evaluator.rescale_to_next_inplace(l2_reg_cipher);
        evaluator.mod_switch_to_inplace(l2_reg_alpha_plain, l2_reg_cipher.parms_id());
        evaluator.multiply_plain_inplace(l2_reg_cipher, l2_reg_alpha_plain);

        evaluator.mod_switch_to_inplace(l2_reg_cipher, delta_beta_all_cipher.parms_id());
        l2_reg_cipher.scale() = pow(2, (int)log2(delta_beta_all_cipher.scale()));
        delta_beta_all_cipher.scale() = pow(2, (int)log2(delta_beta_all_cipher.scale()));
        evaluator.add_inplace(delta_beta_all_cipher, l2_reg_cipher);

        // Test
        decryptor.decrypt(delta_beta_all_cipher, delta_beta_plain);
        ckks_encoder.decode(delta_beta_plain, delta_beta_decode);

        for(int i = 0; i < training_rows; i++) {
            beta[i] = beta[i] - learning_rate * delta_beta_decode[i];
            cout << beta[i] << " ";
        }
        cout << endl;

        // acuracy
        double accuracy = 0.0;
        for(int i = 0; i < testing_rows; i++) {
            double tmp;
            tmp = vector_dot_product(beta, testing_kernel[i]);
            if(tmp >= 0) {
                tmp = 1;
            } else {
                tmp = -1;
            }

            if(tmp == testing_y[i]) accuracy += 1;
        }
        cout << "accuracy " << accuracy / testing_rows << endl;
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << training_rows << " total execution time :\t" << time_diff.count() << " milliseconds" << endl;

    return 0;
}
