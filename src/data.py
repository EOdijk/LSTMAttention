import os
import numpy as np
import re


class Data:
    """
    Contains a .train, .test and .validation (split from .train) set, sorted by entity pairs and relations, where:
    dataset[index][0] = mentions (shape=[tuple_sentences, max_len])
    dataset[index][1] = mentions masks (shape=[tuple_sentences, max_len])
    dataset[index][2] = relation (shape=1D vector, each relation held by the pair)
    dataset[index][3] = entity pair (concatenated string)
    dataset[index][4] = distance to e1 (shape=[tuple_sentences, max_len])
    dataset[index][5] = distance to e2 (shape=[tuple_sentences, max_len])
    Note: the relation dimension (2) is just a list of all relations for that tuple until the
        validation split is done, after which the entries are duplicated for each relation and receive an array
        of size tuple_sentences containing that relation.
        This is to ensure the same-pair-different-label entries stay together
        in either the train or test set when shuffling.

    Alternatively, when split into batches:
    dataset[batch_nr][0] = mentions (shape=[batch_size*tuple_sentences, max_len])
    dataset[batch_nr][1] = mentions masks (shape=[batch_size*tuple_sentences, max_len])
    dataset[batch_nr][2] = relation sentences (shape=batch_size*tuple_sentences)
    dataset[batch_nr][3] = entity pair (shape=[batch_size])
    dataset[batch_nr][4] = distance to e1 (shape=[batch_size*tuple_sentences, max_len])
    dataset[batch_nr][5] = distance to e2 (shape=[batch_size*tuple_sentences, max_len])
    dataset[batch_nr][6] = relation of pair (shape=[batch_size])

    Validation and batch sizes are not saved. Call split_validation and split_batches separately.

    """

    def __init__(self, max_len, tuple_sentences, negatives_portion, word_emb_size, word_emb_pretrained, root_path, batch_size):
        print 'Preprocessing new dataset...'
        self._word_dict = {}  # Dictionary for mapping all words to indices
        self._rel_dict = {}  # Dictionary for mapping all relations to indices
        self._word_dict['UNK'] = 0  # Preset unknown words as index 0 (used for words not available in pre-trained embs)
        self._rel_dict['NA'] = 0  # Preset the negative label as index 0

        self.word_count = 1  # Index for the latest new word. Can be interpreted as the current vocabulary size
        self.rel_count = 1  # Index for the latest new relation. Can be interpreted as the current number of relations

        self.max_len = max_len  # Maximum sentence length
        self.tuple_sentences = tuple_sentences  # Sentences per tuple
        self.negatives_portion = negatives_portion
        self.word_emb_size = word_emb_size
        self.use_pretrained = word_emb_pretrained
        self.batch_size = batch_size
        self._pretrained_words = self._get_pretrained_words()

        self.train = self._preprocess(root_path + '/data/lin/train.txt', False)
        print '  Training set built.'
        self.test = self._preprocess(root_path + '/data/lin/test.txt', True)
        print '  Testing set built.'

        self.pretrained_embeddings = self._get_pretrained_embeddings()

        self.validation_portion = 0
        self.validation = None

    def shuffle(self):
        np.random.seed(10)
        np.random.shuffle(self.train)
        # np.random.shuffle(self.test)
        # if self.validation:
        #    np.random.shuffle(self.validation)

    def split_validation(self, validation_portion):
        """
        Splits off part of the training data to be used as validation data.
        validation_portion should be a float between 0.0 and 1.0
        """

        validation_size = int(round(validation_portion * (len(self.train))))
        self.validation = self.train[:validation_size]
        self.train = self.train[validation_size:]
        self.validation = self.explode_relations(self.validation)

        self.train = self.explode_relations(self.train)
        self.test= self.explode_relations(self.test)

        # self.train = self.validation
        # self.test= self.validation

    def explode_relations(self, set):
        """
        Splits off a new entry for each pair's additional relations.
        Also provides an array so that each sentence has a matching relation.
        """

        # For each entry
        for i in xrange(0, len(set)):

            # If there are multiple relations
            if len(set[i][2]) > 1:
                rel_stack = set[i][2]

                # Create new entries
                for j in xrange(0, len(set[i][2])):
                    rel = rel_stack[j]
                    rel_full = np.full((self.tuple_sentences,), rel, dtype=np.int32)
                    if j == 0:
                        edited = list(set[i])
                        edited[2] = rel_full
                        set[i] = tuple(edited)
                    else:
                        set.append((set[i][0], set[i][1], rel_full, set[i][3], set[i][4], set[i][5],set[i][6], set[i][7]))

            # Otherwise only give the existing entry a full array
            else:
                rel = set[i][2]
                rel_full = np.full((self.tuple_sentences,), rel, dtype=np.int32)
                edited = list(set[i])
                edited[2] = rel_full
                set[i] = tuple(edited)

        return set

    def split_batches(self, batch_size):
        self.train = self.split_list(self.train, batch_size)
        self.test = self.split_list(self.test, batch_size)
        if self.validation:
            self.validation = self.split_list(self.validation, batch_size)

    def split_list(self, list, batch_size):
        new_list = []

        # For each batch
        for i in xrange(0, int(round(len(list) / batch_size))):

            # Combine pairs from the old list
            for j, pair in enumerate((list[i * batch_size:(i + 1) * batch_size])):
                if j == 0:
                    mentions = pair[0]
                    masks = pair[1]
                    relations_sentences = pair[2]
                    tup = pair[3]
                    distances1 = pair[4]
                    distances2 = pair[5]
                    flags1 = pair[6]
                    flags2 = pair[7]
                    relation_pair = pair[2][0]
                else:
                    # Note: could do this faster than one by one
                    mentions = np.vstack((mentions, np.asarray(pair[0], dtype=np.int32)))
                    masks = np.vstack((masks, np.asarray(pair[1], dtype=np.float32)))
                    relations_sentences = np.hstack((relations_sentences, np.asarray(pair[2], dtype=np.int32)))
                    tup = np.vstack((tup, np.asarray(pair[3])))
                    distances1 = np.vstack((distances1, np.asarray(pair[4], dtype=np.int32)))
                    distances2 = np.vstack((distances2, np.asarray(pair[5], dtype=np.int32)))
                    flags1 = np.vstack((flags1, np.asarray(pair[6], dtype=np.int32)))
                    flags2 = np.vstack((flags2, np.asarray(pair[7], dtype=np.int32)))
                    relation_pair = np.hstack((relation_pair, np.asarray(pair[2][0], dtype=np.int32)))

            new_list.append((mentions, masks, relations_sentences, tup, distances1, distances2, relation_pair, flags1, flags2))

        return new_list

    def _get_pretrained_words(self):
        """Get the words in the pre-trained set. Used when pre-processing the corpus."""
        if not self.use_pretrained:
            return None

        root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        emb_path = root_path + '/data/glove.6B.%dd.txt' % self.word_emb_size

        if os.path.exists(emb_path):
            f = open(emb_path)
            pretrained_words = {}
            for line in f:
                content = line.replace('\r', '').replace('\n', '').split(' ')
                word = content[0]
                pretrained_words[word] = True
            return pretrained_words
        else:
            print '  Pretrained embedding path not found!'
            return None

    def _get_pretrained_embeddings(self):
        """Get the pre-trained embeddings, using only the words that appear in the corpus."""
        if not self.use_pretrained:
            return None

        root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        emb_path = root_path + '/data/glove.6B.%dd.txt' % self.word_emb_size

        if os.path.exists(emb_path):
            embeddings = [None] * (self.word_count)
            np.random.seed(10)
            embeddings[0] = np.random.uniform(low=-1.0, high=1.0, size=self.word_emb_size)  # 'UNK' embedding
            f = open(emb_path)
            for line in f:
                content = line.replace('\r', '').replace('\n', '').split(' ')
                word = content[0]
                if word in self._word_dict:
                    index = self._word_dict[word]
                    values = content[1:]
                    embeddings[index] = values
            embeddings = np.asarray(embeddings, dtype=np.float32)
            return embeddings
        else:
            return None


    def _pad_and_mask(self, ment, distance1, distance2, flag1, flag2):
        """Pads the mention with a padding character until size max_len, and creates a corresponding mask"""
        mask = np.zeros(self.max_len, dtype=np.int)
        mask[:len(ment)] = 1
        sentence = np.lib.pad(ment, (0, self.max_len - len(ment)), 'constant', constant_values=0)
        distance1 = np.lib.pad(distance1, (0, self.max_len - len(distance1)), 'constant', constant_values=0)
        distance2 = np.lib.pad(distance2, (0, self.max_len - len(distance2)), 'constant', constant_values=0)
        flag1 = np.lib.pad(flag1, (0, self.max_len - len(flag1)), 'constant', constant_values=2)
        flag2 = np.lib.pad(flag2, (0, self.max_len - len(flag2)), 'constant', constant_values=2)
        return sentence, distance1, distance2, mask, flag1, flag2

    def _pad_matrix(self, matrix):
        """Pads a matrix of mentions or masks with zero vectors until size tuple_sentences"""
        if matrix.ndim == 1:
            size = self.tuple_sentences - 1
        else:
            size = self.tuple_sentences - len(matrix)
        # print size, self.max_len, len(matrix), self.tuple_sentences
        zero = np.zeros((size, self.max_len), dtype=np.int32)
        matrix = np.vstack((matrix, zero))
        return matrix

    def _stuff_to_add(self, matrix):
        """Pads a matrix of mentions or masks with zero vectors until size tuple_sentences"""
        if matrix.ndim == 1:
            size = self.tuple_sentences - 1
        else:
            size = self.tuple_sentences - len(matrix)
        return size

    def _find_positions(self, mention, entity1, entity2):
        """
        Finds the positions of two entities in a mention. Returns -1 if the entity cannot be found
        Only finds the first occurrence of the word.
        """
        position_e1 = -1
        position_e2 = -1
        for i, word in enumerate(mention):
            if word == entity1:
                position_e1 = i
            if word == entity2:
                position_e2 = i

        return position_e1, position_e2

    def _fix_e2(self, mention, entity1, entity2):
        """
        In some cases, entity2 is not concatenated in the mention.
        This is fixed by retrying concatenation, but with the second entity first.
        """
        mention = '_'.join(mention)
        mention = mention.split('_')
        mention = self._concatenate_entity(mention, entity2)
        mention = self._concatenate_entity(mention, entity1)
        return mention

    def _concatenate_entity(self, mention, entity):
        """
        Concatenates an entity in the mention.
        Still does not catch all of them, e.g. if the first word of the entity is the last word of the sentence.
        Those cases are ignored by a try/catch.
        """
        entitysplit = entity.split('_')

        i = 0
        while i < len(mention):
            if mention[i] == entitysplit[0]:
                found = True
                for j in xrange(1, len(entitysplit)):
                    if mention[i+j] != entitysplit[j]:
                        found = False
                        break
                if found:
                    new_mention = []
                    for k in xrange(0, i):
                        new_mention.append(mention[k])
                    new_mention.append(entity)
                    for k in xrange(i + len(entitysplit), len(mention)):
                        new_mention.append(mention[k])
                    mention = new_mention

            i += 1

        return mention

    def _preprocess_mentions(self, entity1, entity2, mention_stack):
        mentions = None
        masks = None
        distances1 = None
        distances2 = None
        flags1 = None
        flags2 = None

        if isinstance(mention_stack, basestring):
            mention_stack = [[mention_stack]]

        # Parse
        for m, ment in enumerate(mention_stack):
            ment = ment[0].split()
            del ment[-1]  # Removing end tag
            length = len(ment)

            if length > self.max_len:
                continue

            # Find positions
            position_e1, position_e2 = self._find_positions(ment, entity1, entity2)
            if position_e2 == -1:
                try:
                    ment = self._fix_e2(ment, entity1, entity2)
                    length = len(ment)  # Fix shortens the length due to concatenation
                    position_e1, position_e2 = self._find_positions(ment, entity1, entity2)
                except:
                    # For the ones that still aren't found
                    return None, None, None, None, None, None

            # If the positions still can't be found, it's a faulty sentence where one of the entities is missing
            # e.g. 'new_york' and 'new_york_city', but only the latter is in the sentence
            if position_e1 == -1 or position_e2 == -1:
                # print "Entity not found in sentence"
                # print "  ", entity1
                # print "  ", entity2
                # print "  ", ment
                return None, None, None, None, None, None

            # Index and pad mentions, create corresponding masks, find distances from entities
            distance1 = np.zeros((length,), dtype=np.int32)
            distance2 = np.zeros((length,), dtype=np.int32)
            flag1 = np.zeros((length,), dtype=np.int32)
            flag2 = np.zeros((length,), dtype=np.int32)

            for i, word in enumerate(ment):
                if word not in self._word_dict:
                    if self.use_pretrained and word not in self._pretrained_words:
                        word = 'UNK'
                    else:
                        self._word_dict[word] = self.word_count
                        self.word_count += 1  # Note: count starts from 1, index starts from 0 ('UNK')
                ment[i] = self._word_dict[word]
                distance1[i] = abs(i - position_e1)
                distance2[i] = abs(i - position_e2)
                flag1[i] = 0 if abs(i - position_e1) == 0 else 1
                flag2[i] = 0 if abs(i - position_e2) == 0 else 1
            ment, distance1, distance2, mask, flag1, flag2 = self._pad_and_mask(ment, distance1, distance2, flag1, flag2)
            ment = np.asarray(ment, dtype=np.int32)
            mask = np.asarray(mask, dtype=np.float32)
            distance1 = np.asarray(distance1, dtype=np.int32)
            distance2 = np.asarray(distance2, dtype=np.int32)
            flag1 = np.asarray(flag1, dtype=np.int32)
            flag2 = np.asarray(flag2, dtype=np.int32)

            # Add to array
            if mentions is None:
                mentions = ment
                masks = mask
                distances1 = distance1
                distances2 = distance2
                flags1 = flag1
                flags2 = flag2
            elif mentions.ndim == 1 or len(mentions) < self.tuple_sentences:  # Note: if a tuple has more sentences, they are discarded
                mentions = np.vstack((mentions, ment))
                masks = np.vstack((masks, mask))
                distances1 = np.vstack((distances1, distance1))
                distances2 = np.vstack((distances2, distance2))
                flags1 = np.vstack((flags1, flag1))
                flags2 = np.vstack((flags2, flag2))

        # Add null sentences/masks until size is self.tuple_sentences
        # stuff_to_add = {}
        if mentions is not None:
            pass
            # stuff_to_add={}
            # mentions = self._pad_matrix(mentions)
            # masks = self._pad_matrix(masks)
            # distances1 = self._pad_matrix(distances1)
            # distances2 = self._pad_matrix(distances2)
            # flags1 = self._pad_matrix(flags1)
            # flags2 = self._pad_matrix(flags2)

            # stuff_to_add['mentions'] = self._stuff_to_add(mentions)
            # stuff_to_add['masks'] = self._stuff_to_add(masks)
            # stuff_to_add['distances1'] = self._stuff_to_add(distances1)
            # stuff_to_add['distances2'] = self._stuff_to_add(distances2)
            # stuff_to_add['flags1'] = self._stuff_to_add(flags1)
            # stuff_to_add['flags2'] = self._stuff_to_add(flags2)

        return mentions, masks, distances1, distances2, flags1, flags2

    def _index_relation(self, rel):
        if rel not in self._rel_dict:
            self.rel_count += 1
            self._rel_dict[rel] = self.rel_count - 1
        rel = self._rel_dict[rel]
        return rel

    def _preprocess(self, path, making_test):
        dataset = []  # Final dataset to be returned
        previous = np.zeros(5)
        mention_stack = None
        rel_stack = None
        rel_pass = 0
        rel_pass_at = int(self.negatives_portion * 100)
        skip = False  # Whether the new mentions found should be saved
        pairs = {}

        with open(path, 'r') as f:
            lines = f.readlines()
        lines.append('-1\t-1\t-1\t-1\t-1\t-1')  # extra dummy line to ensure the last sentence is processed

        for line in lines:
            current = re.split(r'\t', line)

            # If line has the same entities
            if current[2] == previous[2] and current[3] == previous[3]:

                # If line has the same relation, stack up the mentions
                if current[4] == previous[4]:
                    if not skip:
                        mention_stack = np.vstack((mention_stack, current[5]))
                # Otherwise stack up the relations
                else:
                    rel_stack.append(self._index_relation(current[4]))
                    skip = True  # Don't save the next mentions (because they are the same as the ones already found)

            # Otherwise, process the data
            else:
                if mention_stack is not None:
                    pair = previous[2] + "+" + previous[3]
                    if pair not in pairs:  # Duplicates exist in the file. These are ignored
                        pairs[pair] = True
                        mentions, masks, dist1, dist2, f1, f2 = self._preprocess_mentions(previous[2], previous[3], mention_stack)
                        if mentions is not None:  # It's possible that no ments pass the preproc due to being too long
                            rel = self._index_relation(previous[4])  # Note: if it has a neg label, it has no others
                            if rel == 0 and not making_test:  # Drop part of negatives
                                rel_pass += 1  # Not getting bamboozled by np.random again
                                # if rel_pass < rel_pass_at:
                                if np.random.rand() < self.negatives_portion:
                                    # Create new stacks
                                    mention_stack = current[5]
                                    rel_stack = [self._index_relation(current[4])]
                                    skip = False
                                    previous = current
                                    continue
                                else:
                                    rel_pass = 0

                            dataset.append([mentions, masks, rel_stack, pair, dist1, dist2, f1, f2])

                # Create new stacks
                mention_stack = current[5]
                rel_stack = [self._index_relation(current[4])]
                skip = False

            previous = current
        print("unique pairs: ", len(pairs))
        return dataset
