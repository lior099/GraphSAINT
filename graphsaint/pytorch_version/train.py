import copy
import math
import sys
import os

from Code.toy import Toy

sys.path.append(os.path.abspath('GraphSAINT'))
from graphsaint.globals import *
from graphsaint.pytorch_version.models import GraphSAINT
from graphsaint.pytorch_version.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from graphsaint.pytorch_version.utils import *


import torch
import time
from sklearn import metrics


def evaluate_full_batch(model, minibatch, mode='val', toy_title=''):
    """
    Full batch evaluation: for validation and test sets only.
        When calculating the F1 score, we will mask the relevant root nodes
        (e.g., those belonging to the val / test sets).

    UnlinkPrediction edit - also work for train.
    """
    # batch_mode can't be 'train' because we want the full graph, so we change to 'val'
    batch_mode = 'val' if mode == 'train_val_test' else mode
    loss,preds,labels = model.eval_step(*minibatch.one_batch(mode=batch_mode), minibatch)
    if toy_title.split(' ')[-1] in ['1', '10', '50', '100']:
        Toy.draw_line_graph(title=toy_title, node_labels=preds, val_idx=minibatch.node_val)
    # if mode == 'train_val_test':
    #     print()
    if Globals.args_global.loss_type == 'node':
        if mode == 'val':
            target = [minibatch.node_val]
        elif mode == 'test':
            target = [minibatch.node_test]
        elif mode == 'valtest':
            target = [minibatch.node_val, minibatch.node_test]
        else:
            assert mode == 'train_val_test'
            target = [minibatch.node_train, minibatch.node_val, minibatch.node_test]
    elif Globals.args_global.loss_type == 'edge':
        if mode == 'val':
            target = [minibatch.edge_val]
        elif mode == 'test':
            target = [minibatch.edge_test]
        elif mode == 'valtest':
            target = [minibatch.edge_val, minibatch.edge_test]
        else:
            assert mode == 'train_val_test'
            target = [minibatch.edge_train, minibatch.edge_val, minibatch.edge_test]
    f1mic, f1mac = [], []
    auc = []
    for n in target:
        if len(labels[n]):
            f1_scores = calc_f1(to_numpy(labels[n]), to_numpy(preds[n]), model.sigmoid_loss)
            f1mic.append(f1_scores[0])
            f1mac.append(f1_scores[1])
            fpr, tpr, thresholds = metrics.roc_curve(to_numpy(labels[n][:,1]), to_numpy(preds[n]), pos_label=1)
            auc.append(round(metrics.auc(fpr, tpr), 3))
        else:
            f1mic.append(0)
            f1mac.append(0)
            auc.append(0.5)
    f1mic = f1mic[0] if len(f1mic)==1 else f1mic
    f1mac = f1mac[0] if len(f1mac)==1 else f1mac
    auc = auc[0] if len(auc)==1 else auc
    # loss is not very accurate in this case, since loss is also contributed by training nodes
    # on the other hand, for val / test, we mostly care about their accuracy only.
    # so the loss issue is not a problem.
    return loss, f1mic, f1mac, auc, preds



def prepare(train_data,train_params,arch_gcn):
    """
    Prepare some data structure and initialize model / minibatch handler before
    the actual iterative training taking place.
    """
    adj_full, adj_train, feat_full, class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]

    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)
    printf("TOTAL NUM OF PARAMS = {}".format(sum(p.numel() for p in model.parameters())), style="yellow")
    minibatch_eval=Minibatch(adj_full_norm, adj_train, role, train_params, cpu_eval=True)
    model_eval=GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
    if Globals.args_global.gpu >= 0:
        model = model.cuda()
    return model, minibatch, minibatch_eval, model_eval


def train(train_phases, model, minibatch, minibatch_eval, model_eval, eval_val_every, early_stop='loss', name=''):
    if not Globals.args_global.cpu_eval:
        minibatch_eval=minibatch
    epoch_ph_start = 0
    f1mic_best, auc_best, loss_best, ep_best, best_state_dict = 0, 0.5, 100, -1, None
    time_train = 0
    dir_saver = '{}/pytorch_models'.format(Globals.args_global.dir_log)
    path_saver = '{}/pytorch_models/saved_model.pkl'.format(Globals.args_global.dir_log)
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}
    for ip, phase in enumerate(train_phases):
        if phase['end']:
            printf('START PHASE {:4d}'.format(ip),style='underline')
            minibatch.set_sampler(phase)
            num_batches = minibatch.num_training_batches()
        for e in range(epoch_ph_start, int(phase['end'])):
            printf('Epoch {:4d}'.format(e),style='bold')
            minibatch.shuffle()
            l_loss_tr, l_f1mic_tr, l_f1mac_tr = [], [], []
            auc_tr = []
            time_train_ep = 0
            while not minibatch.end():
                t1 = time.time()
                loss_train, preds_train, labels_train = model.train_step(*minibatch.one_batch(mode='train'), minibatch)
                # loss_train, _, _ = model.eval_step(*minibatch.one_batch(mode='true_train'), minibatch)
                time_train_ep += time.time() - t1
                if not minibatch.batch_num % Globals.args_global.eval_train_every:
                    # if not model.sigmoid_loss:
                    #     preds_train = preds_train[:, -1]
                    f1_mic, f1_mac = calc_f1(to_numpy(labels_train),to_numpy(preds_train),model.sigmoid_loss)
                    l_loss_tr.append(loss_train)
                    l_f1mic_tr.append(f1_mic)
                    l_f1mac_tr.append(f1_mac)
                    fpr, tpr, thresholds = metrics.roc_curve(to_numpy(labels_train[:,-1]), to_numpy(preds_train), pos_label=1)
                    auc_tr.append(round(metrics.auc(fpr, tpr), 3))
            if (e+1)%eval_val_every == 0:
                if Globals.args_global.cpu_eval:
                    torch.save(model.state_dict(),'tmp.pkl')
                    model_eval.load_state_dict(torch.load('tmp.pkl',map_location=lambda storage, loc: storage))
                else:
                    model_eval = model
                loss_val, f1mic_val, f1mac_val, auc_val, _ = evaluate_full_batch(model_eval, minibatch_eval, mode='val', toy_title='Line Graph of '+name+' Epoch '+str(e))
                auc_val = take_worse(Globals.f_mean(auc_tr), auc_val)
                # printf(' TRAIN (Ep avg): loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\ttrain time = {:.4f} sec'\
                #         .format(Globals.f_mean(l_loss_tr), Globals.f_mean(l_f1mic_tr), Globals.f_mean(l_f1mac_tr), time_train_ep))
                # printf(' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}'\
                #         .format(loss_val, f1mic_val, f1mac_val), style='yellow')
                printf(f'train auc = {Globals.f_mean(auc_tr)}')
                printf(f'validation auc = {auc_val}', style='yellow')

                new_values = [Globals.f_mean(l_loss_tr), Globals.f_mean(auc_tr), loss_val, auc_val]

                # loss, f1mic, f1mac, auc = evaluate_full_batch(model_eval, minibatch_eval,
                #                                                         mode='train_val_test')
                # # loss_train, loss_val, loss_test = loss
                # f1mic_train, f1mic_val, f1mic_test = f1mic
                # f1mac_train, f1mac_val, f1mac_test = f1mac
                # auc_train, auc_val, auc_test = auc
                # auc_val = min(auc_train, auc_val)
                #
                # printf(' TRAIN (Ep avg): loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\ttrain time = {:.4f} sec' \
                #        .format(-1, f1mic_train, f1mac_train, time_train_ep))
                # printf(' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}' \
                #        .format(-1, f1mic_val, f1mac_val), style='yellow')
                # print('train auc = ', auc_train)
                # print('validation auc = ', auc_val)
                # new_values = [loss_train, auc_train, loss_val, auc_val]

                history = {key: values + [new_value] for (key, values), new_value in zip(history.items(), new_values)}

                if e == 0 or (early_stop == 'auc' and auc_val > auc_best) or (early_stop == 'f1' and f1mic_val > f1mic_best) or (early_stop == 'loss' and loss_val < loss_best):
                    f1mic_best, auc_best, loss_best, ep_best = f1mic_val, auc_val, loss_val, e
                    if not os.path.exists(dir_saver):
                        os.makedirs(dir_saver)
                    printf('  Saving state_dict ...', style='yellow')
                    best_state_dict = copy.deepcopy(model.state_dict())
            time_train += time_train_ep
            if e - ep_best >= 50:
                break
        if best_state_dict:
            printf('  Saving model ...', style='yellow')
            torch.save(best_state_dict, path_saver)
        epoch_ph_start = int(phase['end'])
    printf("Optimization Finished!", style="yellow")
    if ep_best >= 0:
        if Globals.args_global.cpu_eval:
            model_eval.load_state_dict(torch.load(path_saver, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(path_saver))
            model_eval=model
        printf('  Restoring model ...', style='yellow')
    loss, f1mic_both, f1mac_both, auc, preds = evaluate_full_batch(model_eval, minibatch_eval, mode='train_val_test')
    f1mic_train, f1mic_val, f1mic_test = f1mic_both
    f1mac_train, f1mac_val, f1mac_test = f1mac_both
    auc_train, auc_val, auc_test = auc
    auc_val = take_worse(auc_train, auc_val)

    printf("Full validation (Epoch {:4d}): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}"\
            .format(ep_best, f1mic_val, f1mac_val), style='red')
    printf("Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}"\
            .format(f1mic_test, f1mac_test), style='red')
    print('Final train auc = ', auc_train)
    print('Final validation auc = ', auc_val)
    print('Final test auc = ', auc_test)
    printf("Total training time: {:6.2f} sec".format(time_train), style='red')
    return {'train': auc_train, 'val': auc_val, 'test': auc_test}, history, preds


def take_worse(auc_train, auc_val):
    train_value = abs(auc_train-0.5)
    val_value = abs(auc_val-0.5)
    if train_value < val_value:
        print("replaced auc_val: "+str(auc_val)+" with auc_val: "+str(auc_train))
        return auc_train
    return auc_val

def save_state(auc_dict, history, preds):
    import pickle
    with open(Globals.args_global.dir_log+'/state.pkl', 'wb') as file:
        pickle.dump((Globals.args_global, Globals.timestamp, auc_dict, history, preds), file, protocol=pickle.HIGHEST_PROTOCOL)

def start_gs_train(gs_args, graph_name):
    print("Starting GraphSAINT!")
    import warnings
    warnings.filterwarnings("ignore")
    Globals.update_globals(gs_args)
    # open_model()
    # os.chdir('C:/Users/shifmal2/Documents/Pycharm Projects/GraphSAINT/GraphSAINT-master')
    # print('args_global:', Globals.args_global)
    log_dir(Globals.args_global.train_config, Globals.args_global.data_prefix, Globals.git_branch, Globals.git_rev, Globals.timestamp)
    train_params, train_phases, train_data, arch_gcn = parse_n_prepare(Globals.args_global)
    if 'eval_val_every' not in train_params:
        train_params['eval_val_every'] = Globals.EVAL_VAL_EVERY_EP
    model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)

    if Globals.args_global.saved_model_path:
        # state_dict = torch.load(Globals.args_global.saved_model_path, map_location=lambda storage, loc: storage)
        #
        # model.load_state_dict(state_dict)
        try:
            if Globals.args_global.cpu_eval:
                model_eval.load_state_dict(torch.load(Globals.args_global.saved_model_path, map_location=lambda storage, loc: storage))
            else:
                model.load_state_dict(torch.load(Globals.args_global.saved_model_path))
                model_eval = model
        except:
            raise Exception("Error when loading model from last snapshot. Maybe all tags are 0?")
    auc_dict, history, preds = train(train_phases, model, minibatch, minibatch_eval, model_eval, train_params['eval_val_every'], name=graph_name)
    save_state(auc_dict, history, preds)


if __name__ == '__main__':
    pass
    # print("Starting GraphSAINT!")
    # import warnings
    # warnings.filterwarnings("ignore")
    # # open_model()
    # # os.chdir('C:/Users/shifmal2/Documents/Pycharm Projects/GraphSAINT/GraphSAINT-master')
    # log_dir(Globals.args_global.train_config, Globals.args_global.data_prefix, Globals.git_branch, Globals.git_rev, Globals.timestamp)
    # train_params, train_phases, train_data, arch_gcn = parse_n_prepare(Globals.args_global)
    # if 'eval_val_every' not in train_params:
    #     train_params['eval_val_every'] = Globals.EVAL_VAL_EVERY_EP
    # model, minibatch, minibatch_eval, model_eval = prepare(train_data, train_params, arch_gcn)
    #
    # if Globals.args_global.saved_model_path:
    #     state_dict = torch.load(Globals.args_global.saved_model_path, map_location=lambda storage, loc: storage)
    #     model.load_state_dict(state_dict)
    #     model_eval.load_state_dict(state_dict)
    # auc_dict = train(train_phases, model, minibatch, minibatch_eval, model_eval, train_params['eval_val_every'])
    # save_state(auc_dict)
