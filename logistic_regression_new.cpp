#include <omp.h>
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
    // int poly_modulus_degree = 32768;
    // int poly_modulus_degree = 16384;
    int poly_modulus_degree = 8192;
    EncryptionParameters params(scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    cout << "MAX BIT COUNT: " << CoeffModulus::MaxBitCount(poly_modulus_degree) << endl;
    // params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 35, 35, 35, 35, 35, 35, 35, 35, 35, 60}));
    params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {40, 25, 25, 25, 25, 25, 40}));
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
    // double scale = pow(2.0, 35);
    double scale = pow(2.0, 25);

    // Time
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::milliseconds time_diff;

    // Read File
    // string filename = "./python/datasets/breast_cancer/data.csv";
    string filename = "./python/datasets/make_circles.csv";
    vector<vector<string>> s_matrix = CSVtoMatrix(filename);
    vector<vector<double>> f_matrix = stringTodoubleMatrix(s_matrix);
    // random_shuffle(f_matrix.begin(), f_matrix.end());

    // Init features, labels and weights
    int total_rows = f_matrix.size();
    cout << "\nNumber of rows  = " << total_rows << endl;
    int total_cols = f_matrix[0].size() - 1;
    cout << "\nNumber of cols  = " << total_cols << endl;

    vector<vector<double>> features(total_rows, vector<double>(total_cols));

    int training_rows = 400;
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
    vector<double> weights(total_cols);
    for (int i = 0; i < total_cols; i++) {
        weights[i] = 0;
        // weights[i] = RandomFloat(-0.1, 0.1);
    }

    // Polynomial Simulated Sigmoid Function
    double poly_deg = 3;
    // double poly_deg = 7;

    vector<double> coeffs = {0.50014, 0.01404, -0.00000007, -0.000001};
    // vector<double> coeffs = {0.50054, 0.19688, -0.00014, -0.00544, 0.000005, 0.000075, -0.00000004, -0.0000003};


    // Parameters Settings
    int col_A = 1;
    int col_B = total_cols - col_A;

    double learning_rate = 0.001;
    int iter_times = 20;

    vector<vector<double>> features_A(training_rows, vector<double>(col_A));
    vector<vector<double>> features_B(training_rows, vector<double>(col_B));

    // calculate features_A
    for(int i = 0; i < training_rows; i++) {
        for (int j = 0; j < col_A; j++) { 
            features_A[i][j] = training_x[i][j];
        }
    }

    // calculate features_B
    for(int i = 0; i < training_rows; i++) {
        for (int j = 0; j < col_B; j++) { 
            features_B[i][j] = training_x[i][col_A + j];
        }
    }

    // Calculate gradient descents in the encrypted domain

    // --------------- ENCODING ----------------
    time_start = chrono::high_resolution_clock::now();

    cout << "ENCODING......\n";
    vector<Plaintext> features_A_plain(training_rows), features_B_plain(training_rows);

#pragma omp parallel for
    for (int i = 0; i < training_rows; i++) {
        ckks_encoder.encode(features_A[i], scale, features_A_plain[i]);
        ckks_encoder.encode(features_B[i], scale, features_B_plain[i]);
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << training_rows << " total encoding time :\t" << time_diff.count() << " milliseconds" << endl;

    // --------------- ENCRYPTNG ------------------
    time_start = chrono::high_resolution_clock::now();

    cout << "ENCRYPTING......\n";
    vector<Ciphertext> features_A_cipher(training_rows), features_B_cipher(training_rows);
#pragma omp parallel for
    for (int i = 0; i < training_rows; i++) {
        encryptor.encrypt(features_A_plain[i], features_A_cipher[i]);
        encryptor.encrypt(features_B_plain[i], features_B_cipher[i]);
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << training_rows << " total encryption time :\t" << time_diff.count() << " milliseconds" << endl;
   
    // --------------- CALCULATTNG ------------------
    time_start = chrono::high_resolution_clock::now();

    cout << "CALCULATING......\n";
    vector<Ciphertext> features_cipher(training_rows);  // x
#pragma omp parallel for
    for(int i = 0; i < training_rows; i++) {
        evaluator.rotate_vector(features_B_cipher[i], -col_A, gal_keys, features_cipher[i]);
        evaluator.add_inplace(features_cipher[i], features_A_cipher[i]);
    }

    double one = 1;
    Plaintext one_plain;
    Ciphertext one_cipher;
    ckks_encoder.encode(one, scale, one_plain);
    encryptor.encrypt(one_plain, one_cipher);
    
    double alpha;
    Plaintext alpha_plain, weights_plain;
    Ciphertext x_cipher, weights_cipher, weights_features_cipher, delta_w_all_cipher;
    vector<Ciphertext> delta_w_cipher(training_rows);
    vector<Ciphertext> wx_powers_cipher(poly_deg);
    // used when decoding
    Plaintext delta_w_plain;
    vector<double> delta_w_decode(total_cols);

    for(int iter = 0; iter < iter_times; iter++) {
        cout << "iter " << iter << endl;
        ckks_encoder.encode(weights, scale, weights_plain);
        encryptor.encrypt(weights_plain, weights_cipher);
// #pragma omp parallel for
        for(int i = 0; i < training_rows; i++) {
            x_cipher = features_cipher[i];

            weights_features_cipher = cipher_dot_product(x_cipher, weights_cipher, total_cols, relin_keys, gal_keys, evaluator);

            compute_all_powers(weights_features_cipher, poly_deg, evaluator, relin_keys, wx_powers_cipher);
            wx_powers_cipher[0] = one_cipher;

            int last_id = wx_powers_cipher.size() - 1;
            parms_id_type last_parms_id = wx_powers_cipher[last_id].parms_id();
            double last_scale = pow(2, (int)log2(wx_powers_cipher[last_id].scale()));

            for(int j = 0; j <= poly_deg; j++) {
                evaluator.mod_switch_to_inplace(wx_powers_cipher[j], last_parms_id);
                wx_powers_cipher[j].scale() = last_scale;
            }

            for(int j = 0; j <= poly_deg; j++) {
                alpha = coeffs[j] * pow(-1 * training_y[i], j + 1) / training_rows;
                ckks_encoder.encode(alpha, scale, alpha_plain);
                evaluator.mod_switch_to_inplace(alpha_plain, wx_powers_cipher[j].parms_id());
                evaluator.multiply_plain_inplace(wx_powers_cipher[j], alpha_plain);
                // evaluator.rescale_to_next_inplace(wx_powers_cipher[j]);
            }

            evaluator.add_many(wx_powers_cipher, delta_w_cipher[i]);
            evaluator.mod_switch_to_inplace(x_cipher, delta_w_cipher[i].parms_id());
            evaluator.multiply_inplace(delta_w_cipher[i], x_cipher);
        }

        evaluator.add_many(delta_w_cipher, delta_w_all_cipher);

        // Test
        decryptor.decrypt(delta_w_all_cipher, delta_w_plain);
        ckks_encoder.decode(delta_w_plain, delta_w_decode);

        for(int i = 0; i < total_cols; i++) {
            weights[i] = weights[i] - learning_rate * delta_w_decode[i];
            cout << weights[i] << " ";
        }
        cout << endl;

        // acuracy
        double accuracy = 0.0;
        for(int i = 0; i < testing_rows; i++) {
            double tmp;
            tmp = vector_dot_product(weights, testing_x[i]);
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
