#include <iostream>
#include <iomanip>
#include <fstream>
#include <unistd.h>
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

// Matrix Transpose
vector<vector<double>> transpose_matrix(vector<vector<double>> input_matrix)
{

    int rowSize = input_matrix.size();
    int colSize = input_matrix[0].size();
    vector<vector<double>> transposed(colSize, vector<double>(rowSize));

    for (int i = 0; i < rowSize; i++)
    {
        for (int j = 0; j < colSize; j++)
        {
            transposed[j][i] = input_matrix[i][j];
        }
    }

    return transposed;
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


void Matrix_Multiplication(size_t poly_modulus_degree)
{
    // read file
    string filename = "pulsar_stars.csv";
    vector<vector<string>> s_matrix = CSVtoMatrix(filename);
    vector<vector<double>> f_matrix = stringTodoubleMatrix(s_matrix);

    // Init features, labels and weights
    // Init features (rows of f_matrix , cols of f_matrix - 1)
    int rows = f_matrix.size();
    rows = 1000;
    cout << "\nNumber of rows  = " << rows << endl;
    int cols = f_matrix[0].size() - 1;
    cout << "\nNumber of cols  = " << cols << endl;

    vector<vector<double>> features(rows, vector<double>(cols));
    // Init labels (rows of f_matrix)
    vector<double> labels(rows);
    // Init weight vector with zeros (cols of features)
    vector<double> weights(rows);

    // Fill the features matrix and labels vector
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            features[i][j] = f_matrix[i][j];
        }
        labels[i] = f_matrix[i][cols];
    }

    // Fill the weights with random numbers (from 1 - 2)
    for (int i = 0; i < rows; i++)
    {
        weights[i] = RandomFloat(-2, 2) + 0.00000001;
        // cout << "weights[i] = " << weights[i] << endl;
    }

    vector<vector<double>> standard_features = standard_scaler(features);

    double lambda = 0.01;
    
    int col_A = 3;
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
                    standard_features[i][col_B + k] * standard_features[j][col_B + k] + 0.00000001;
            }
        }
    }

    vector<vector<double>> kernel_A_diagonals(rows, vector<double>(rows));
    vector<vector<double>> kernel_B_diagonals(rows, vector<double>(rows));
    // vector<vector<double>> kernel_diagonals(rows, vector<double>(rows));

    for (int i = 0; i < rows; i++) {
        kernel_A_diagonals[i] = get_diagonal(i, kernel_A);
        kernel_B_diagonals[i] = get_diagonal(i, kernel_B);
    }

    /*
    print_partial_matrix(kernel_A);
    print_partial_matrix(kernel_A_diagonals);
    print_partial_matrix(kernel_B);
    print_partial_matrix(kernel_B_diagonals);
    */

    // Combine two kernels together
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) { 
            kernel[i][j] = kernel_A[i][j] + kernel_B[i][j];
        }
    }

    // Calculate Gradient vector
    vector<double> gradient_A = linear_transformation(kernel, weights);
    vector<double> gradient_B = linear_transformation(kernel, weights);

    for(int i = 0; i < rows; i++) {
        gradient_B[i] = gradient_A[i] / (4 * rows) - labels[i] / ( 2 * rows);
        gradient_A[i] = gradient_A[i] * 2 * lambda / rows;
    }

    vector<double> gradient_C = linear_transformation(kernel, gradient_B);

    for(int i = 0; i < rows; i++) {
        gradient_A[i] = gradient_A[i] + gradient_C[i];
    }

    // Test print first 10 rows
    cout << "\nFirst 10 rows of gradient --------\n" << endl;
    for (int i = 0; i < 10; i++) {
            cout << gradient_A[i] << ", ";
    }
    cout << endl;
    cout << "\nLast 10 rows of gradient ----------\n" << endl;
    // Test print last 10 rows
    for (int i = gradient_A.size() - 10; i < gradient_A.size(); i++)
    {
            cout << gradient_A[i] << ", ";
    }
    cout << endl;

    // Handle Rotation Error First
    int dimension = 4;
    if (dimension > poly_modulus_degree / 4)
    {
        cerr << "Dimension is too large. Choose a dimension less than " << poly_modulus_degree / 4 << endl;
        exit(1);
    }

    EncryptionParameters params(scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    cout << "MAX BIT COUNT: " << CoeffModulus::MaxBitCount(poly_modulus_degree) << endl;
    params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 35, 35, 35, 35, 35, 35, 35, 60}));
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
    
    // First Experiments add two kernels together.

    // --------------- ENCODING ----------------
    auto start_time = chrono::high_resolution_clock::now();
    cout << "ENCODING......\n";
    vector<Plaintext> kernel_A_plain(rows), kernel_B_plain(rows);
    Plaintext weights_plain, labels_plain;
    for (int i = 0; i < rows; i++) {
        ckks_encoder.encode(kernel_A_diagonals[i], scale, kernel_A_plain[i]);
        ckks_encoder.encode(kernel_B_diagonals[i], scale, kernel_B_plain[i]);
    }
    ckks_encoder.encode(weights, scale, weights_plain); // may not be used
    ckks_encoder.encode(labels, scale, labels_plain); // may not be used
    auto stop_time = chrono::high_resolution_clock::now();
    auto duration_time = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
    cout << "\nTime to Encode" << duration_time.count() << " microseconds" << endl;


    // --------------- ENCRYPTNG ------------------
    start_time = chrono::high_resolution_clock::now();
    cout << "ENCRYPTING......\n";
    vector<Ciphertext> kernel_A_cipher(rows), kernel_B_cipher(rows);
    Ciphertext weights_cipher, labels_cipher;
    for (int i = 0; i < rows; i++) {
        encryptor.encrypt(kernel_A_plain[i], kernel_A_cipher[i]);
        encryptor.encrypt(kernel_B_plain[i], kernel_B_cipher[i]);
    }
    encryptor.encrypt(weights_plain, weights_cipher);
    encryptor.encrypt(labels_plain, labels_cipher);
    stop_time = chrono::high_resolution_clock::now();
    duration_time = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
    cout << "\nTime to Encrypt" << duration_time.count() << " microseconds" << endl;

    // Add together
    cout << "COMBINING KERNELS TOGETHER......\n";
    for(int i = 0; i < rows; i++) {
        evaluator.add_inplace(kernel_A_cipher[i], kernel_B_cipher[i]);
    }

    // Calculate Gradients

    // Define constants
    double tmp_1 = 2.0 * lambda / rows;
    double tmp_2 = 1.0 / (4 * rows);
    double tmp_3 = 1.0 / (2 * rows);
    cout << "tmp_1 = " << tmp_1 << endl;
    cout << "tmp_2 = " << tmp_2 << endl;
    cout << "tmp_3 = " << tmp_3 << endl;
    Plaintext tmp_1_plain, tmp_2_plain, tmp_3_plain;
    ckks_encoder.encode(tmp_1, scale, tmp_1_plain);
    ckks_encoder.encode(tmp_2, scale, tmp_2_plain);
    ckks_encoder.encode(tmp_3, scale, tmp_3_plain);

    start_time = chrono::high_resolution_clock::now();
    Ciphertext gradient_1_cipher, gradient_2_cipher, gradient_3_cipher, gradient_4_cipher;

    // Linear_Transform_CipherMatrix_PlainVector has an issue that vector doesn't rotate in each iterations.
    // gradient_1_cipher = Linear_Transform_CipherMatrix_PlainVector(weights_plain, kernel_A_cipher, gal_keys, params);
    gradient_1_cipher = Linear_Transform_Cipher(weights_cipher, kernel_A_cipher, gal_keys, params);
    
    evaluator.rescale_to_next_inplace(gradient_1_cipher);
    evaluator.relinearize_inplace(gradient_1_cipher, relin_keys);

    parms_id_type gradient_1_cipher_parms_id = gradient_1_cipher.parms_id();
    evaluator.mod_switch_to_inplace(tmp_1_plain, gradient_1_cipher_parms_id);
    evaluator.mod_switch_to_inplace(tmp_2_plain, gradient_1_cipher_parms_id);

    cout << tmp_1_plain.parms_id() << endl;
    cout << log2(tmp_1_plain.scale()) << endl;

    evaluator.multiply_plain(gradient_1_cipher, tmp_1_plain, gradient_2_cipher);
    evaluator.multiply_plain(gradient_1_cipher, tmp_2_plain, gradient_3_cipher);
    evaluator.multiply_plain_inplace(labels_cipher, tmp_3_plain);

    evaluator.mod_switch_to_next_inplace(labels_cipher);

    gradient_3_cipher.scale() = pow(2, (int)log2(gradient_3_cipher.scale()));
    labels_cipher.scale() = pow(2, (int)log2(labels_cipher.scale()));

    evaluator.sub_inplace(gradient_3_cipher, labels_cipher);

    for(int i = 0; i < kernel_A_cipher.size(); i++) {
        evaluator.mod_switch_to_next_inplace(kernel_A_cipher[i]);
    }

    gradient_4_cipher = Linear_Transform_Cipher(gradient_3_cipher, kernel_A_cipher, gal_keys, params);

    evaluator.rescale_to_next_inplace(gradient_4_cipher);
    evaluator.relinearize_inplace(gradient_4_cipher, relin_keys);

    evaluator.mod_switch_to_inplace(gradient_2_cipher, gradient_4_cipher.parms_id());

    gradient_2_cipher.scale() = pow(2, (int)log2(gradient_2_cipher.scale()));
    gradient_4_cipher.scale() = pow(2, (int)log2(gradient_4_cipher.scale()));

    cout << gradient_4_cipher.parms_id() << endl;
    cout << log2(gradient_4_cipher.scale()) << endl;

    cout << gradient_2_cipher.parms_id() << endl;
    cout << log2(gradient_2_cipher.scale()) << endl;

    evaluator.add_inplace(gradient_2_cipher, gradient_4_cipher);
    stop_time = chrono::high_resolution_clock::now();
    duration_time = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
    cout << "\nTime to Calculate Gradient Descents" << duration_time.count() << " microseconds" << endl;


    cout << "BEGIN TO DECRYPT \n";

    // Decrypt and Decode 

    // Test Gradient 2 results
    start_time = chrono::high_resolution_clock::now();
    Plaintext gradient_2_plain;
    vector<double> gradient_2_decode(rows);

    decryptor.decrypt(gradient_2_cipher, gradient_2_plain);
    ckks_encoder.decode(gradient_2_plain, gradient_2_decode);
    stop_time = chrono::high_resolution_clock::now();
    duration_time = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
    cout << "\nTime to Decrypt and Decode" << duration_time.count() << " microseconds" << endl;

    // Test print first 10 rows
    cout << "\nFirst 10 rows of kernels --------\n" << endl;
    for (int i = 0; i < 10; i++) {
            cout << gradient_A[i] << ", " << gradient_2_decode[i] << endl;
    }
    cout << endl;


    /*
    vector<Plaintext> kernel_plain(rows);
    vector<vector<double>> kernel_decode(rows, vector<double>(rows));
    for(int i = 0; i < rows; i++) {
        decryptor.decrypt(kernel_A_cipher[i], kernel_plain[i]);
        ckks_encoder.decode(kernel_plain[i], kernel_decode[i]);
    }

    // Test print first 10 rows
    cout << "\nFirst 10 rows of kernels --------\n" << endl;
    for (int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            cout << i << " " << j << " " << kernel[i][j] << ", " << kernel_decode[i][j] << endl;
        }
        cout << endl;
    }
    cout << endl;
    */


    cout << "Done" << endl;
    

}

int main()
{

    Matrix_Multiplication(16384);

    return 0;
}
