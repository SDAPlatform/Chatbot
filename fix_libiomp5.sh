rm ./venv/lib/python3.12/site-packages/torch/.dylibs/libiomp5.dylib
rm ./venv/lib/python3.12/site-packages/torch/lib/libiomp5.dylib
ln -s ../../ctranslate2/.dylibs/libiomp5.dylib ./venv/lib/python3.12/site-packages/torch/.dylibs/libiomp5.dylib
ln -s ../../ctranslate2/.dylibs/libiomp5.dylib ./venv/lib/python3.12/site-packages/torch/lib/libiomp5.dylib
