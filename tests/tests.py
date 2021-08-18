#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""Tests module
@author: Aymen Jemi (jemix) <jemiaymen@gmail.com> at SUDO-AI

Tests tools for sudoai.
"""

import unittest

from sudoai.dataset import DataType, DatasetType, DatasetInfo, Dataset, DatasetError

from sudoai.utils import load_tokenizer


class DatasetTest(unittest.TestCase):

    def test_dataset_info(self):
        info = DatasetInfo(id='sa',
                           data_path='data.txt',
                           data_type=DataType.TEXT)
        self.assertEqual(info.data_type, DataType.TEXT)

    def test_dataset(self):
        info = DatasetInfo(id='sa', data_path='data.txt',
                           dataset_type=DatasetType.WORD_TO_LABEL)

        with self.assertRaises(FileNotFoundError):
            tokenizer = load_tokenizer('sa')

            with self.assertRaises(DatasetError):
                dataset = Dataset(info=info, tokenizer=tokenizer)
                dataset.build()


class UtilsTest(unittest.TestCase):
    pass


class TokenizerTest(unittest.TestCase):
    pass


class TrainerTest(unittest.TestCase):
    pass


class PipelineTest(unittest.TestCase):
    pass


class HypertuningTest(unittest.TestCase):
    pass


# class TestText(unittest.TestCase):

#     def test_is_arabic(self):
#         word_arabic = 'شبيك'
#         t = Text()
#         self.assertTrue(t.is_arabic(word_arabic),
#                         "test method : is_arabic DONE")

#     def test_clean_test(self):
#         t = Text()
#         line = "chbik @ # % yakhi , ..    aymen"
#         line = t.clean_text(line, ascii=True)
#         verify = "chbik yakhi aymen"
#         self.assertEqual(line, verify, "test method : clean_test DONE")

#     def test_call_method(self):
#         line = "chbik ¨^@ # % yakhi , ..    aymen"
#         verify = "chbik yakhi aymen"
#         t = Text()
#         line = t(line)
#         self.assertEqual(
#             line, verify, "test method : __call__ with param DONE")

#     def test_unique_chars(self):
#         words = ["aymen", "winek", "lyouma"]
#         t = Text()
#         self.assertEqual(len(t.unique_chars(words)), 11)

#     def test_wc(self):
#         path = '.test'
#         t = Text()
#         self.assertEqual(t.wc(path), 511)

#     def test_wc_both(self):
#         path = '.test'
#         t = Text()
#         self.assertEquals(t.wc(path, both=True), (511, 26))


# class TestCharTokenizer(unittest.TestCase):

#     def test_call_or_encoder_method(self):
#         path = '.test'
#         data = 'سبعين'
#         tokens = [[29, 18, 35, 46, 42]]
#         tk = CharTokenizer()
#         tk.train(path)
#         self.assertEquals(tokens, tk(data))

#     def test_decoder_method(self):
#         path = '.test'
#         data = 'سبعين'
#         tokens = [[29, 18, 35, 46, 42]]
#         tk = CharTokenizer()
#         tk.train(path)
#         self.assertEquals(data, tk.decoder(tokens, is_word=False))

#     def test_non_trained_chartk(self):
#         with self.assertRaises(NotTrainedError):
#             data = 'سبعين'
#             tk = CharTokenizer()
#             tk(data)

#     def test_input_type_error(self):
#         with self.assertRaises(InputTypeError):
#             # data = 'سبعين'
#             data = 10
#             path = '.test'
#             tk = CharTokenizer()
#             tk.train(path)
#             tk(data)

#     def test_encode_error(self):
#         with self.assertRaises(InputTypeError):
#             data = 'سبعين'
#             path = '.test'
#             tk = CharTokenizer()
#             tk.train(path)
#             tk.decoder(data)

#     def test_save_tk(self):
#         path = '.test'
#         tk = CharTokenizer()
#         tk.train(path)

#         r = tk.save('test.tk')
#         self.assertTrue(r)

#     def test_load_tk(self):
#         path = '.test'
#         tk = CharTokenizer()
#         tk.train(path)

#         tk2 = CharTokenizer()

#         tk2 = tk2.load('test.tk')
#         self.assertEqual(tk, tk2)

#     def test_new_train_tk(self):
#         path = '.testar'
#         data = 'njarbou'
#         tokens = [[23, 19, 10, 27, 11, 24, 30]]
#         tk = CharTokenizer()
#         tk.train(path)
#         self.assertEqual(tokens, tk(data))


# class TestTokenizer(unittest.TestCase):

#     def test_non_trained_tk(self):
#         with self.assertRaises(NotTrainedError):
#             tk = Tokenizer()

#             print(tk(['njarbou hadha']))

#     def test_train_tk(self):
#         tk = Tokenizer('.testar')

#         en = tk('njarbou hadha')
#         test = [104205, 53246]

#         self.assertEqual(en, test)

#     def test_decode_tk(self):
#         tk = Tokenizer('.testar')

#         en = [104205, 53246]
#         de = 'njarbou hadha'

#         test = tk.decoder(en)

#         self.assertEqual(de, test)

#     def test_save_tk(self):

#         tk = Tokenizer('.testar')
#         r = tk.save('tk_nonch.tk')

#         self.assertTrue(r)

#     def test_load_tk(self):
#         tk = Tokenizer('.testar')

#         tk2 = tk.load('tk_nonch.tk')

#         self.assertEqual(tk, tk2)

#     def test_load_tk_2(self):
#         tk = Tokenizer()
#         tk = tk.load('tk_nonch.tk')

#         en = [[104205, 53246]]
#         de = ['njarbou hadha']

#         test = tk.decoder(en, is_line=False)

#         self.assertEqual(de, test)

#     def test_vocab_size(self):
#         tk = Tokenizer()
#         tk = tk.load('tk_nonch.tk')

#         self.assertEqual(tk.vocab_size, 154531)


# class TestDatasets(unittest.TestCase):

#     def test_dataset_vs_multi_label_dataset(self):
#         tk = FastTokenizer(
#             "D:\\projects\\sudo-ai\\sudo-ai\\data\\labels.test", 1)

#         label_to_index = {'good': [0, 1], 'bad': [1, 0]}
#         index_to_label = {1: 'good', 0: 'bad'}

#         info = DatasetInfo(name='tn',
#                            version='0.1.0',
#                            sep='|',
#                            data_path="data/labels.test",
#                            label2index=label_to_index,
#                            index2label=index_to_label,
#                            do_cleaning=True,
#                            data_type=DataType.TEXT)
#         data = Dataset(info, tk)
#         data2 = MultiLabelDataset(info, tk)
#         return self.assertEqual(len(data), len(data2))

#     def test_dataset_vs_multi_class_dataset(self):
#         tk = FastTokenizer(
#             "D:\\projects\\sudo-ai\\sudo-ai\\data\\labels.test", 1)

#         label_to_index = {'good': [0, 1], 'bad': [1, 0]}
#         index_to_label = {1: 'good', 0: 'bad'}

#         info = DatasetInfo(name='tn',
#                            version='0.1.0',
#                            sep='|',
#                            data_path="data/labels.test",
#                            label2index=label_to_index,
#                            index2label=index_to_label,
#                            do_cleaning=True,
#                            data_type=DataType.TEXT)
#         data = Dataset(info, tk)
#         data2 = MultiClassDataset(info, tk)
#         return self.assertEqual(len(data), len(data2))

#     def test_word2word_dataset(self):
#         # just lower case lettre
#         src = FastCharTokenizer(1)
#         # arabic letter
#         target = FastCharTokenizer(2)

#         info = DatasetInfo(name='tn',
#                            version='0.1.0',
#                            data_path="data/ttd.xlsx",
#                            do_cleaning=True,
#                            data_type=DataType.EXCEL)
#         data = Word2WordDataset(info, src, target)

#         return self.assertEqual(data.len(), (40, 47))
#     def test_seq2seq_dataset(self):
#         src = FastTokenizer("data/ar.csv", 1)
#         target = FastTokenizer("data/tn.csv", 2)
#         info = DatasetInfo(name='tn',
#                            version='0.1.0',
#                            data_path="data/ttd.csv",
#                            sep=',',
#                            data_type=DataType.TEXT)
#         data = Seq2SeqDataset(info, src, target)
#         return self.assertEqual(data.len(), (24, 20))
if __name__ == '__main__':
    unittest.main()
