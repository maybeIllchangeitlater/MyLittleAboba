all: install

install: uninstall
	mkdir build
	mkdir $(HOME)/Desktop/MLP
	cd build && cmake ../CMakeLists.txt && make
	mv build/MultilayerAbobatron.app $(HOME)/Desktop/MLP/MLP.app

uninstall: 
	rm -rf build
	rm -rf $(HOME)/Desktop/MLP

dist: uninstall install
	mkdir dist
	cp -r $(HOME)/Desktop/MLP/dist/.
	cp ../README.md dist
	rm -rf dist

dvi:
	doxygen Doxyfile
	open ./html/index.html/

.PHONY: install uninstall dist dvi
	

