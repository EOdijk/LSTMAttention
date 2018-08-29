import os
import cPickle as pickle
import matplotlib.pyplot as plt


def append_values(x_values, y_values, file, graphtype, epoch):
    if graphtype == 'AUC':
        y_values.append(file.pr_area_list)
        x_values.append(range(0, len(file.pr_area_list)))
    elif graphtype == 'F1':
        finals_list = []
        for i in xrange(0, len(file.f1_measure_list)):
            final = len(file.f1_measure_list[i]) - 1
            finals_list.append(file.f1_measure_list[i][final])
        y_values.append(finals_list)
        x_values.append(range(0, len(finals_list)))
    elif graphtype == 'Precision':
        finals_list = []
        for i in xrange(0, len(file.precision_list)):
            final = len(file.precision_list[i]) - 1
            finals_list.append(file.precision_list[i][final])
        y_values.append(finals_list)
        x_values.append(range(0, len(finals_list)))
    elif graphtype == 'Recall':
        finals_list = []
        for i in xrange(0, len(file.recall_list)):
            final = len(file.recall_list[i]) - 1
            finals_list.append(file.recall_list[i][final])
        y_values.append(finals_list)
        x_values.append(range(0, len(finals_list)))
    elif graphtype == 'PR':
        if epoch == -1:
            epoch = file.best_epoch # epochs start numbering from 1
        x_values.append(file.recall_list[epoch - 1])
        y_values.append(file.precision_list[epoch - 1])
    elif graphtype == 'PR_Test':
        x_values.append(file.recall_test)
        y_values.append(file.precision_test)
    return x_values, y_values


def print_stats(file):
    best_epoch = file.best_epoch

    print '  Train:'
    print '    For best epoch: ', best_epoch
    print '    AUC: ', file.best_pr_area
    # print '    AUC CHECK: ', file.pr_area_list[best_epoch - 1]  # Note: same as best_pr_area
    # print '    Precision: ', file.precision_list[best_epoch - 1]
    # print '    Recall: ', file.recall_list[best_epoch - 1]
    # print '    F1: ', file.f1_measure_list[best_epoch - 1]

    print '  Test:'
    print '    AUC: ', file.pr_area_test
    # print '    Precision: ', file.precision_test
    # print '    Recall: ', file.recall_test
    # print '    F1: ', file.f1_measure_test


def label_accuracies(stats_path, epoch=-1):
    """
    Produce a table of accuracies for each label in the given stats file at the given epoch.
    If epoch is not set or under 0, test results are shown instead.
    """
    file = pickle.load(open(stats_path, 'rb'))
    if epoch < 0:
        predictions = file.predictions_test
        truths = file.truths_test
        name = "test"
    else:
        predictions = file.predictions_val[epoch]
        truths = file.truths_val[epoch]
        name = "validation"

    num_labels = (max(truths) + 1)
    total_count = len(truths)  # Number of entries in the validation/test set

    count_per_label = [0] * num_labels  # Amount of times the label was predicted
    pos_per_label = [0] * num_labels  # Amount of time the label was predicted accurately
    truths_per_label = [0] * num_labels  # Amount of golds for that label

    # Process stats
    print("Processing...")
    for i in xrange(0, total_count):
        predicted_label = int(predictions[i])
        true_label = int(truths[i])

        count_per_label[predicted_label] += 1
        truths_per_label[true_label] += 1
        if predicted_label == true_label:
            pos_per_label[predicted_label] += 1

    # Print results
    print "Label accuracies for ", stats_path, name
    print "Label - Total predicted - Correct predictions - Total for label - Precision - Recall"
    for l in xrange(0, num_labels):
        count = count_per_label[l]
        correct = pos_per_label[l]
        total = truths_per_label[l]
        precision = 0.0 if count == 0 else round(float(correct) / float(count), 4)
        recall = 0.0 if total == 0 else round(float(correct) / float(total), 4)
        print l, "&", count, "&", correct, "&", total, "&", precision, "&", recall, "\\\\"


def get_results(path, graphtype, epoch=-1):
    """
    For each stats file in the given directory, produce a graph as determined by graphtype.
    If an epoch number is given, PR will yield results for that epoch rather than the best.
    Graphtypes: Precision, Recall, AUC, F1, PR, PR_Test
    """
    x_values = []
    y_values = []
    names = []

    #  Read stats.pk files
    for filename in os.listdir(path):
        if filename.endswith('.pk'):
            print 'Analyzing file: ' + filename
            file = pickle.load(open(os.path.join(path, filename), 'rb'))
            # print_stats(file)
            names.append(os.path.basename(filename)[:-3])
            x_values, y_values = append_values(x_values, y_values, file, graphtype, epoch)

    #  Create graphs
    for i in xrange(0, len(y_values)):
        plt.plot(x_values[i], y_values[i], label=names[i])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc=2)
    #plt.title(graphtype) # + ' ' + str(epoch))
    plt.title('Precision-Recall on Test Set for Word Attention')
    #plt.title('AUC for Word Attention')
    if graphtype == 'PR' or graphtype == 'PR_Test':
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        axes = plt.gca()
        axes.set_xlim([0.0, 0.5])
        axes.set_ylim([0.0, 1.0])
    else:
        plt.xlabel('Epoch')
        plt.ylabel(graphtype)
        axes = plt.gca()
        axes.set_ylim([0.0, 0.25])
    #plt.show()
    #plt.savefig(saved_path + '/pr' + str(epoch), bbox_inches='tight')
    plt.savefig(saved_path + '/wa_pr_test', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    saved_path = os.path.join(root_path, 'saved')
    get_results(saved_path, 'PR_Test')
    #label_accuracies(os.path.join(saved_path, 'LSTM.pk'))
