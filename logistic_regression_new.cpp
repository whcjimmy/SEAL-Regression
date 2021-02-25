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
    // int poly_modulus_degree = 32768;
    int poly_modulus_degree = 16384;
    // int poly_modulus_degree = 8192;
    EncryptionParameters params(scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    cout << "MAX BIT COUNT: " << CoeffModulus::MaxBitCount(poly_modulus_degree) << endl;
    // params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 35, 35, 35, 35, 35, 60}));
    params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 35, 35, 35, 35, 35, 35, 35, 35, 35, 60}));
    // params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {40, 25, 25, 25, 25, 25, 40}));
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
    double scale = pow(2.0, 35);
    // double scale = pow(2.0, 25);

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
    int total_rows = f_matrix.size();
    int rows = f_matrix.size();
    cout << "\nNumber of rows  = " << rows << endl;
    int cols = f_matrix[0].size() - 1;
    cout << "\nNumber of cols  = " << cols << endl;

    vector<vector<double>> features(rows, vector<double>(cols));
    // Init labels (rows of f_matrix)
    vector<double> labels(rows);
    // Init weight vector with zeros (cols of features)
    vector<double> weights(cols);

    // Fill the features matrix and labels vector
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            features[i][j] = f_matrix[i][j];
        }
        labels[i] = f_matrix[i][cols];
    }

    // Fill the weights with random numbers (from 1 - 2)
    for (int i = 0; i < cols; i++) {
        weights[i] = RandomFloat(0.0001, 0.0005);
        // weights[i] = RandomFloat(-2, 2) + 0.00000001;
    }

    vector<vector<double>> standard_features = minmax_scaler(features);

    // seperate features into two parts
    rows = 736;
    int col_A = 17;
    int col_B = cols - col_A;

    vector<vector<double>> features_A(rows, vector<double>(col_A));
    vector<vector<double>> features_B(rows, vector<double>(col_B));

    // calculate features_A
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < col_A; j++) { 
                features_A[i][j] = standard_features[i][j];
        }
    }

    // calculate features_B
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < col_B; j++) { 
                features_B[i][j] = standard_features[i][col_A + j];
        }
    }

    double lambda = 0.01;
    double poly_deg = 3;
    // double poly_deg = 7;

    vector<double> coeffs = {0.50091, 0.19832, -0.00018, -0.00447};
    // vector<double> coeffs = {0.50040, 0.22933, -0.00022, -0.01026, 0.00001, 0.00021};
    // vector<double> coeffs = {0.50016, 0.24218, -0.00017, -0.01488, 0.00002, 0.000061, -0.0000009, -0.0000003};
    double learning_rate = 0.01;
    int iter_times = 10;
    
    // Calculate gradient descents in the plaintext domain
    //
    vector<double> weights_1 = weights;
    vector<double> delta_w(cols, 0.0);
    double w_x, tmp;

    for(int iter = 0; iter < iter_times; iter++) {
        for(int i = 0; i < rows; i++) {
            w_x = vector_dot_product(weights_1, standard_features[i]);
            tmp = 0.0;

            for(int j = 0; j < poly_deg; j++) {
                tmp = coeffs[j] * pow(-1 * labels[i], j + 1) / rows;
                tmp = tmp * pow(w_x, j);
                for(int k = 0; k < cols; k++) {
                    delta_w[k] += tmp * standard_features[i][k];
                }
            }
        }

        for(int i = 0; i < cols; i++) {
            weights_1[i] = weights_1[i] - learning_rate * delta_w[i];
        }

        // output results
        cout << "iter " << iter << endl;
        for(int i = 0; i < cols; i++) {
            cout << weights_1[i] << " ";
        }
        cout << endl;
    }


    // Calculate gradient descents in the encrypted domain

    // --------------- ENCODING ----------------
    time_start = chrono::high_resolution_clock::now();

    cout << "ENCODING......\n";
    vector<Plaintext> features_A_plain(rows), features_B_plain(rows);

    for (int i = 0; i < rows; i++) {
        ckks_encoder.encode(features_A[i], scale, features_A_plain[i]);
        ckks_encoder.encode(features_B[i], scale, features_B_plain[i]);
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << rows << " total encoding time :\t" << time_diff.count() << " milliseconds" << endl;

    // --------------- ENCRYPTNG ------------------
    time_start = chrono::high_resolution_clock::now();

    cout << "ENCRYPTING......\n";
    vector<Ciphertext> features_A_cipher(rows), features_B_cipher(rows);
    for (int i = 0; i < rows; i++) {
        encryptor.encrypt(features_A_plain[i], features_A_cipher[i]);
        encryptor.encrypt(features_B_plain[i], features_B_cipher[i]);
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << rows << " total encryption time :\t" << time_diff.count() << " milliseconds" << endl;
   
    // --------------- CALCULATTNG ------------------
    time_start = chrono::high_resolution_clock::now();

    cout << "CALCULATING......\n";
    vector<Ciphertext> features_cipher(rows);  // x
    for(int i = 0; i < rows; i++) {
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
    Ciphertext x_cipher, weights_features_cipher, delta_w_all_cipher;
    vector<Ciphertext> delta_w_cipher(rows);
    vector<Ciphertext> wx_powers_cipher(poly_deg);
    // used when decoding
    Plaintext delta_w_plain;
    vector<double> delta_w_decode(cols);

    for(int iter = 0; iter < iter_times; iter++) {
        cout << "iter " << iter << endl;
        ckks_encoder.encode(weights, scale, weights_plain);
        for(int i = 0; i < rows; i++) {
            x_cipher = features_cipher[i];
            evaluator.multiply_plain(x_cipher, weights_plain, weights_features_cipher);
            evaluator.rescale_to_next_inplace(weights_features_cipher);

            compute_all_powers(weights_features_cipher, poly_deg, evaluator, relin_keys, wx_powers_cipher);
            wx_powers_cipher[0] = one_cipher;

            for(int j = 0; j <= poly_deg; j++) {
                alpha = coeffs[j] * pow(-1 * labels[i], j + 1) / rows;
                ckks_encoder.encode(alpha, scale, alpha_plain);
                
                evaluator.mod_switch_to_inplace(alpha_plain, wx_powers_cipher[j].parms_id());
                evaluator.multiply_plain_inplace(wx_powers_cipher[j], alpha_plain);
                evaluator.rescale_to_next_inplace(wx_powers_cipher[j]);

                evaluator.mod_switch_to_inplace(x_cipher, wx_powers_cipher[j].parms_id());
                evaluator.multiply_inplace(wx_powers_cipher[j], x_cipher);
            }

            int last_id = wx_powers_cipher.size() - 1;
            parms_id_type last_parms_id = wx_powers_cipher[last_id].parms_id();
            double last_scale = pow(2, (int)log2(wx_powers_cipher[last_id].scale()));

            for(int j = 0; j <= poly_deg; j++) {
                evaluator.mod_switch_to_inplace(wx_powers_cipher[j], last_parms_id);
                wx_powers_cipher[j].scale() = last_scale;
            }

            evaluator.add_many(wx_powers_cipher, delta_w_cipher[i]);
        }

        evaluator.add_many(delta_w_cipher, delta_w_all_cipher);


        // Test
        decryptor.decrypt(delta_w_all_cipher, delta_w_plain);
        ckks_encoder.decode(delta_w_plain, delta_w_decode);

        for(int i = 0; i < cols; i++) {
            weights[i] = weights[i] - learning_rate * delta_w_decode[i];
            cout << weights[i] << " ";
        }
        cout << endl;
    }

    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << rows << " total execution time :\t" << time_diff.count() << " milliseconds" << endl;

    // acuracy
    double acc_1 = 0.0, acc_2 = 0.0;
    for(int i = 736; i < total_rows; i++) {
        double tmp_1, tmp_2;
        tmp_1 = vector_dot_product(weights_1, standard_features[i]);
        tmp_2 = vector_dot_product(weights, standard_features[i]);
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
