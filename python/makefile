execute:
	POVM_DEBUG=5 POVM_EXPERIMENTAL=True python shgo/main.py
	# POVM_DEBUG=5 POVM_EXPERIMENTAL=True python gradient_descent/main.py
	# POVM_DEBUG=5 POVM_EXPERIMENTAL=False python shgo/main.py

read:
	# POVM_EXPERIMENTAL=True python read_results.py
	POVM_EXPERIMENTAL=False python read_results.py

test:
	pytest tests/

documentation:
	# clean old docs
	rm doc/api/*
	$(MAKE) -C doc clean
	# make new docs
	sphinx-apidoc -o doc/api/ .
	$(MAKE) -C doc html
