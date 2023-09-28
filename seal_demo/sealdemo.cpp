#include "seal/seal.h"
#include <iostream>
#include <stdint.h>

using namespace std;
using namespace seal;

typedef uint64_t u64;

vector<int64_t> center_lift(Plaintext plaintext, u64 size, u64 modulus) {
	vector<int64_t> ret(size);
	
	for (int i = 0; i < plaintext.coeff_count(); i++) {
		ret[i] = (plaintext[i] > modulus/2) ? plaintext[i]-modulus : plaintext[i];
	}

	return ret;
}

void zero_vector(vector<int64_t> &vec, u64 size) {
	for (u64 i = 0; i < size; i++) {
		vec[i] = 0;
	}
}

int main() {
	EncryptionParameters parms(scheme_type::bfv);
	size_t poly_modulus_degree = 4096;
	parms.set_poly_modulus_degree(poly_modulus_degree);

	parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));

	parms.set_plain_modulus(1024);

	SEALContext context(parms);

	KeyGenerator keygen(context);
	SecretKey secret_key = keygen.secret_key();
	PublicKey public_key;
	keygen.create_public_key(public_key);

	Encryptor encryptor(context, public_key);

	Evaluator evaluator(context);

	Decryptor decryptor(context, secret_key);
	
	// uint64_t x = 6;
	// uint64_t *x = (uint64_t*) malloc( sizeof(uint64_t) * 10 );
	vector<uint64_t> x;
	cout << "x: ";
	for (int i = 0; i < 10; i++) {
		x.push_back(i);
		cout << i << " ";
	}
	cout << "\n";

	vector<uint64_t> y;
	cout << "y: ";
	for (int i = 0; i < 3; i++) {
		y.push_back(-i-1);
		cout << -i-1 << " ";
	}
	cout << "\n";

	// cout << util::uint_to_dec_string(x, std::size_t(10)) << endl;

	Plaintext x_plain(x, std::size_t(10));
	Plaintext y_plain(y, std::size_t(3));

	Ciphertext x_encrypted;
	encryptor.encrypt(x_plain, x_encrypted);

	evaluator.multiply_plain_inplace(x_encrypted, y_plain);

	Plaintext x_decrypted;
	decryptor.decrypt(x_encrypted, x_decrypted);

	vector<int64_t> lifted = center_lift(x_decrypted, 16, 1024);
	cout << "center_lifted output: ";
	for (auto i: lifted) {
		cout << i << " ";
	} cout << endl;

	zero_vector(lifted, 16);
	for (auto i: lifted) {
		cout << i << " ";
	} cout << endl;

	cout << "0x" << x_decrypted.to_string() << " ..... Correct." << endl;


	return 0;
}
