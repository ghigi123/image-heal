from struct import pack, unpack, Struct
from os import path
import torch

class VectorWriter:
    def __init__(self, filename, vector_size):
        self._filename = filename
        self._vector_size = vector_size
        self._vector_pack = Struct(f'{vector_size}f').pack
        self._file = None
        self._default_id = 0
        self._ids = None

    def __enter__(self):
        self._file = open(f'{self._filename}.db.binary', 'wb')
        self._ids = open(f'{self._filename}.db.txt', 'w')
        self._file.write(pack('I', self._vector_size))
        return self

    def write_batch(self, batch_vectors, batch_ids=None):
        if batch_ids is None:
            id_start = self._default_id
            self._default_id += len(batch_vectors)
            batch_ids = range(id_start, self._default_id)

        for vector in batch_vectors:
            self._file.write(self._vector_pack(*vector))

        self._ids.write('\n'.join(str(id_) for id_ in batch_ids) + '\n')

    def write(self, vector, id=None):
        if id is None:
            self._ids.write(f'{self._default_id}\n')
            self._default_id += 1
        self._file.write(self._vector_pack(*vector))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()
        self._ids.close()


class VectorReader:
    def __init__(self, filename):
        self._filename = filename
        self._vector_size = None
        self._vector_unpack = None
        self._file = None
        self._n = None
        self.ids = None

    def __enter__(self):
        self._file = open(f'{self._filename}.db.binary', 'rb')
        with open(f'{self._filename}.db.txt', 'r') as f:
            self.ids = [id_.strip() for id_ in f.readlines()]

        self._vector_size, = unpack('I', self._file.read(4))
        self._vector_unpack = Struct(f'{self._vector_size}f').unpack
        self._n = (path.getsize(f'{self._filename}.db.binary') - 4) // (4 * self._vector_size)

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

    def hashes(self):
        self._file.seek(4)
        for i in range(len(self)):
            yield self._file.read(4 * self._vector_size)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()

    def __len__(self):
        return self._n


if __name__ == '__main__':
    import torch
    from time import time
    with VectorWriter('out/test', 40) as vw:
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

    with VectorReader('out/test') as vr:
        v0 = vr.read(0)
        print(vr.read(3))
        v0b = vr.read(0)
        assert all(v0[i] == v0b[i] for i in range(len(v0)))
        print(vr[2:5])
        print(vr.ids[:50])