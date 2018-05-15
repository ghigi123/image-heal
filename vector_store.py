from struct import pack, unpack, Struct
from os import path
import torch

class VectorWriter:
    def __init__(self, filename, vector_size):
        self._filename = filename
        self._vector_size = vector_size
        self._vector_pack = Struct(f'{vector_size}f').pack
        self._file = None

    def __enter__(self):
        self._file = open(self._filename, 'wb')
        self._file.write(pack('I', self._vector_size))
        return self

    def write_batch(self, batch):
        write = self.write
        for vector in batch:
            write(vector)

    def write(self, vector):
        self._file.write(self._vector_pack(*vector))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()


class VectorReader:
    def __init__(self, filename):
        self._filename = filename
        self._vector_size = None
        self._vector_unpack = None
        self._file = None
        self._n = None

    def __enter__(self):
        self._file = open(self._filename, 'rb')

        self._vector_size, = unpack('I', self._file.read(4))
        self._vector_unpack = Struct(f'{self._vector_size}f').unpack
        self._n = (path.getsize(self._filename) - 4) // (4 * self._vector_size)

        return self

    def read(self, i):
        real_i = i * 4 * self._vector_size + 4
        self._file.seek(real_i)
        return torch.Tensor(self._vector_unpack(self._file.read(4 * self._vector_size)))

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, end, step = item.indices(self._n)
            # TODO : Careful step is not handled
            self._file.seek(start * 4 * self._vector_size + 4)
            return torch.Tensor([self._vector_unpack(self._file.read(4 * self._vector_size)) for i in range(end - start)])
        else:
            return self.read(item)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()


if __name__ == '__main__':
    import torch
    from time import time
    with VectorWriter('test.test', 40) as vw:
        tensor = torch.randn(40)
        print(tensor)
        vw.write(tensor)
        tensor = torch.randn(500, 40)
        print(tensor[2])
        t = time()
        n = 50
        for i in range(n):
            vw.write_batch(tensor)
        print((time() - t) / n)

    with VectorReader('test.test') as vr:
        v0 = vr.read(0)
        print(vr.read(3))
        v0b = vr.read(0)
        assert all(v0[i] == v0b[i] for i in range(len(v0)))
        print(vr[2:5])
