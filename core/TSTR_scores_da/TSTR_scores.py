import numpy as np
from sklearn.model_selection import train_test_split

from core.TSTR_scores_da.utils import get_data, get_dataloader, save_scores, save_cm
from core.TSTR_scores_da.train_functions import train_cv, train_only, evaluate_model, train_and_test, fine_tune

config = {
    'realworld': {
        'dataset_name': 'realworld',
        'num_df_domains': 10,
        'num_dp_domains': 5,
        'num_classes': 4,
        'class_names': ['WAL', 'RUN', 'CLD', 'CLU'],
        'num_timesteps': 128,
        'num_channels': 3,
        'num_classes': 4,
    },
    'cwru': {
        'dataset_name': 'cwru_256_3ch_5cl',
        'num_df_domains': 4,
        'num_dp_domains': 4,
        'num_classes': 5,
        'class_names': ['IR', 'Ball', 'OR_centred', 'OR_orthogonal', 'OR_opposite'],
        'num_timesteps': 256,
        'num_channels': 3,
        'num_classes': 5,
    },
    'realworld_mobiact': {
        'dataset_name': 'realworld_mobiact',
        'num_df_domains': 15,
        'num_dp_domains': 61,
        'num_classes': 4,
        'class_names': ['WAL', 'RUN', 'CLD', 'CLU'],
        'num_timesteps': 128,
        'num_channels': 3,
        'num_classes': 4,
    },
    'mobiact_realworld': {
        'dataset_name': 'mobiact_realworld',
        'num_df_domains': 61,
        'num_dp_domains': 15,
        'num_classes': 4,
        'class_names': ['WAL', 'RUN', 'CLD', 'CLU'],
        'num_timesteps': 128,
        'num_channels': 3,
        'num_classes': 4,
    }
}



def compute_TSTR_Dp(dataset, num_epochs):
    name = 'Dp'

    accs = []
    f1s = []
    total_cm = None

    for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
        # if dataset == 'realworld_mobiact' and domain == 74:
        #     continue
        print(f"Domain: {domain}")

        # Load Dp data
        x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', domain)
        # x_dp_dom, _, y_dp_dom, _ = train_test_split(x_dp_dom, y_dp_dom, train_size=0.5, stratify=y_dp_dom, shuffle=True, random_state=seed)
        # print("Warning: Using only small fraction of Dp data")
        print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}')

        # Train and evaluate via cross-validation on Dp data
        print('Training and evaluating on Dp data via cross-validation...')
        loss, acc, f1, cm = train_cv(x_dp_dom, y_dp_dom, num_epochs)
        save_scores(domain, loss, acc, f1, name, dataset)
        accs.append(acc)
        f1s.append(f1)
        print(f'Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')
        if total_cm is None:
            total_cm = cm
        else:
            total_cm += cm

    print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")
    save_cm(total_cm, name, dataset)



def compute_TSTR_Df(dataset, num_epochs, num_runs, augment=False):
    name = 'Df_aug' if augment else 'Df'

    accs = []
    f1s = []
    total_cm = None

    for i in range(num_runs):
        accs_run = []
        f1s_run = []

        # Load Df data
        x_df, y_df, k_df = get_data(dataset, 'df')
        # x_df, _, y_df, _ = train_test_split(x_df, y_df, train_size=0.1, stratify=y_df, shuffle=True, random_state=seed)
        # print("Warning: Using only small fraction of Df data")
        print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)} | np.unique(k_df): {np.unique(k_df)}')

        # Train on Df data
        print('Training on Df data...')
        df_model = train_only(x_df, y_df, num_epochs, augment)

        for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
            print(f"Domain: {domain}")

            # Load Dp data
            x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp_te', domain)
            print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}')

            # Evaluate on Dp data
            loss, acc, f1, cm = evaluate_model(df_model, get_dataloader(x_dp_dom, y_dp_dom))
            save_scores(domain, loss, acc, f1, name, dataset)
            accs_run.append(acc)
            f1s_run.append(f1)
            print(f'Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')
            if total_cm is None:
                total_cm = cm
            else:
                total_cm += cm

        print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f} +- {np.std(accs_run):.4f}")
        print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f} +- {np.std(f1s_run):.4f}\n\n")
        accs.append(np.mean(accs_run))
        f1s.append(np.mean(f1s_run))

    print(f"Mean accuracy over {num_runs} runs: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1 over {num_runs} runs: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")
    total_cm = total_cm / num_runs
    save_cm(total_cm, name, dataset)




def compute_TSTR_Syn(dataset, syn_name, num_epochs, num_runs):

    accs = []
    f1s = []
    total_cm = None

    for i in range(num_runs):

        accs_run = []
        f1s_run = []

        for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
            print(f"Domain: {domain}")

            # Load synthetic data
            x_syn_dom, y_syn_dom, k_syn_dom = get_data(dataset, syn_name, domain)
            # x_syn_dom, _, y_syn_dom, _ = train_test_split(x_syn_dom, y_syn_dom, train_size=0.01, stratify=y_syn_dom, shuffle=True)
            # print("Warning: Using only small fraction of Syn data")
            print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)} | np.unique(k_syn_dom): {np.unique(k_syn_dom)}')

            # Load Dp data
            x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp_te', domain)
            print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}')

            # Train on synthetic data and evaluate on Dp data
            print('Training on synthetic data...')
            loss, acc, f1, cm = train_and_test(x_syn_dom, y_syn_dom, x_dp_dom, y_dp_dom, num_epochs)
            save_scores(domain, loss, acc, f1, syn_name, dataset)
            accs_run.append(acc)
            f1s_run.append(f1)
            print(f'Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')
            if total_cm is None:
                total_cm = cm
            else:
                total_cm += cm

        print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f} +- {np.std(accs_run):.4f}")
        print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f} +- {np.std(f1s_run):.4f}\n\n")
        accs.append(np.mean(accs_run))
        f1s.append(np.mean(f1s_run))

    print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")
    total_cm = total_cm / num_runs
    save_cm(total_cm, syn_name, dataset)




def compute_TSTR_Df2Syn(dataset, syn_name, num_epochs, num_runs, augment=False):
    name = f'Df2{syn_name}_aug' if augment else f'Df2{syn_name}'

    accs = []
    f1s = []
    total_cm = None

    for i in range(num_runs):

        accs_run = []
        f1s_run = []

        # Load Df data
        x_df, y_df, k_df = get_data(dataset, 'df')
        # x_df, _, y_df, _ = train_test_split(x_df, y_df, train_size=0.01, stratify=y_df, shuffle=True)
        # print("Warning: Using only small fraction of Df data")
        print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)} | np.unique(k_df): {np.unique(k_df)}')

        # Train on Df data
        print('Training on Df data...')
        df_model = train_only(x_df, y_df, num_epochs, augment)

        for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
            print(f"Domain: {domain}")

            # Load synthetic data
            x_syn_dom, y_syn_dom, k_syn_dom = get_data(dataset, syn_name, domain)
            # x_syn_dom, _, y_syn_dom, _ = train_test_split(x_syn_dom, y_syn_dom, train_size=0.1, stratify=y_syn_dom, shuffle=True, random_state=seed)
            # print("Warning: Using only small fraction of Syn data")
            print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)} | np.unique(k_syn_dom): {np.unique(k_syn_dom)}')

            # Load Dp data
            x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp_te', domain)
            print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}')

            # Fine-tune on synthetic data and evaluate on Dp data
            print('Fine-tuning on synthetic data...')
            finetuned_model = fine_tune(df_model, x_syn_dom, y_syn_dom, num_epochs)
            loss, acc, f1, cm = evaluate_model(finetuned_model, get_dataloader(x_dp_dom, y_dp_dom))
            save_scores(domain, loss, acc, f1, name, dataset)
            accs_run.append(acc)
            f1s_run.append(f1)
            print(f'Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')
            if total_cm is None:
                total_cm = cm
            else:
                total_cm += cm

        print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f} +- {np.std(accs_run):.4f}")
        print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f} +- {np.std(f1s_run):.4f}\n\n")
        accs.append(np.mean(accs_run))
        f1s.append(np.mean(f1s_run))

    print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")
    total_cm = total_cm / num_runs
    save_cm(total_cm, name, dataset)



def compute_TSTR_CORAL(dataset, num_epochs, num_runs, augment=False, coral_weight=1):
    name = 'CORAL_aug' if augment else 'CORAL'

    accs = []
    f1s = []
    total_cm = None

    for i in range(num_runs):

        accs_run = []
        f1s_run = []

        # Load Df data
        x_df, y_df, k_df = get_data(dataset, 'df')
        # x_df, _, y_df, _ = train_test_split(x_df, y_df, train_size=0.1, stratify=y_df, shuffle=True, random_state=2710)
        # print("Warning: Using only small fraction of Df data")
        print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)} | np.unique(k_df): {np.unique(k_df)}\n')

        for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
            print(f"Domain: {domain}")

            # Load Dp map data
            x_dp_map, y_dp_map, k_dp_map = get_data(dataset, 'dp_map', domain)
            print(f'x_dp_map.shape: {x_dp_map.shape} | np.unique(y_dp_map): {np.unique(y_dp_map)} | np.unique(k_dp_map): {np.unique(k_dp_map)}')

            # Load Dp test data
            x_dp_te, y_dp_te, k_dp_te = get_data(dataset, 'dp_te', domain)
            print(f'x_dp_te.shape: {x_dp_te.shape} | np.unique(y_dp_te): {np.unique(y_dp_te)} | np.unique(k_dp_te): {np.unique(k_dp_te)}')

            # Train on Df data and Dp data with CORAL and evaluate on Dp data
            print('Training on Df data and Dp data with CORAL...')
            loss, acc, f1, cm = train_and_test(x_df, y_df, x_dp_te, y_dp_te, num_epochs, augment=augment, coral_weight=coral_weight, coral_train=x_dp_map)
            save_scores(domain, loss, acc, f1, name, dataset)
            accs_run.append(acc)
            f1s_run.append(f1)
            print(f'Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')
            if total_cm is None:
                total_cm = cm
            else:
                total_cm += cm

        print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f} +- {np.std(accs_run):.4f}")
        print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f} +- {np.std(f1s_run):.4f}\n\n")
        accs.append(np.mean(accs_run))
        f1s.append(np.mean(f1s_run))

    print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")
    total_cm = total_cm / num_runs
    save_cm(total_cm, name, dataset)



def compute_TSTR_PL(dataset):
    raise ValueError
    accs = []
    f1s = []

    for i in range(num_runs):

        accs_run = []
        f1s_run = []

        for src_class in config[dataset]['class_names']:
            if src_class != 'WAL':
                continue

            print(f"Source class: {src_class}\n")

            # Load Df data
            x_df, y_df, k_df = get_data(dataset, 'df', src_class)
            # x_df, _, y_df, _ = train_test_split(x_df, y_df, train_size=0.5, stratify=y_df, shuffle=True, random_state=seed)
            # print("Warning: Using only small fraction of Df data")
            print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)} | np.unique(k_df): {np.unique(k_df)}\n')

            # Train on Df data
            print('Training on Df data...')
            df_model = train_only(x_df, y_df, dataset, num_epochs=num_epochs, patience=patience)

            for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
                print(f"Domain: {domain}")

                # Load Dp data
                x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
                print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)} | np.unique(k_dp_dom): {np.unique(k_dp_dom)}')

                # Evaluate on Dp data with pseudo-labeling
                loss, acc, f1 = pseudo_labeling(df_model, x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs, patience=patience)
                save_scores(src_class, domain, loss, acc, f1, 'PL', dataset)
                accs_run.append(acc)
                f1s_run.append(f1)
                print(f'Source class: {src_class} | Domain: {domain} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}\n')

        print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f} +- {np.std(accs_run):.4f}")
        print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f} +- {np.std(f1s_run):.4f}\n\n")
        accs.append(np.mean(accs_run))
        f1s.append(np.mean(f1s_run))

    print(f"Mean accuracy over {num_runs} runs: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print(f"Mean F1 over {num_runs} runs: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")



