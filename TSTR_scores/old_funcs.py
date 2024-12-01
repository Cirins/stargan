

# def compute_TSTR_Dp(dataset):
#     accs = []
#     f1s = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             if domain == 74:
#               continue

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Train and evaluate via cross-validation on Dp data
#             print('Training and evaluating on Dp data via cross-validation...')
#             acc, loss, f1 = train_classifier_cv(x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs)
#             save_scores(src_class, domain, acc, loss, f1, 'Dp', dataset)
#             accs.append(acc)
#             f1s.append(f1)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f} | F1: {f1:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
#     print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")



# def compute_TSTR_Df_rot(dataset):
#     accs = []
#     f1s = []

#     for i in range(num_runs):

#         accs_run = []
#         f1s_run = []

#         for src_class in config[dataset]['class_names']:
#             if src_class != 'WAL':
#                 continue

#             print(f"Source class: {src_class}\n")

#             # Load Df data
#             x_df, y_df, k_df = get_data(dataset, 'df', src_class, rot=True)
#             print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)}\n')

#             # Train on Df data
#             print('Training on Df data...')
#             df_model = train_only(x_df, y_df, dataset, num_epochs=num_epochs)

#             for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#                 print(f"Domain: {domain}")

#                 # Load Dp data
#                 x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#                 print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#                 # Evaluate on Dp data
#                 acc, loss, f1 = evaluate_model(df_model, get_dataloader(x_dp_dom, y_dp_dom))
#                 save_scores(src_class, domain, acc, loss, f1, 'Df_rot', dataset)
#                 accs_run.append(acc)
#                 f1s_run.append(f1)
#                 print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f} | F1: {f1:.4f}\n')

#         print(f"Mean accuracy for run {i}: {np.mean(accs_run):.4f}")
#         print(f"Mean F1 for run {i}: {np.mean(f1s_run):.4f}\n")
#         accs.append(np.mean(accs_run))
#         f1s.append(np.mean(f1s_run))

#     print(f"Mean accuracy over {num_runs} runs: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
#     print(f"Mean F1 over {num_runs} runs: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")




# def compute_TSTR_Syn_aug(dataset):
#     accs = []
#     f1s = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             # Load synthetic data
#             x_syn_dom, y_syn_dom, k_syn_dom = get_syn_data(dataset, syn_name, src_class, domain)
#             print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             y0_indices_syn = np.where(y_syn_dom == 0)[0]
#             y0_indices_dp = np.where(y_dp_dom == 0)[0]
#             assert np.array_equal(y0_indices_syn, y0_indices_dp), "Labels do not match"
#             assert np.array_equal(y_syn_dom[y0_indices_syn], y_dp_dom[y0_indices_dp]), "Labels do not match"

#             y0_indices_train, y0_indices_test = train_test_split(y0_indices_syn, test_size=0.5, shuffle=True, random_state=seed)

#             x_syn_dom_y0, y_syn_dom_y0 = x_syn_dom[y0_indices_train], y_syn_dom[y0_indices_train]
#             x_dp_dom_y0, y_dp_dom_y0 = x_dp_dom[y0_indices_test], y_dp_dom[y0_indices_test]

#             x_syn_dom = np.concatenate([x_syn_dom_y0, x_syn_dom[y_syn_dom != 0]])
#             y_syn_dom = np.concatenate([y_syn_dom_y0, y_syn_dom[y_syn_dom != 0]])
#             x_dp_dom = np.concatenate([x_dp_dom_y0, x_dp_dom[y_dp_dom != 0]])
#             y_dp_dom = np.concatenate([y_dp_dom_y0, y_dp_dom[y_dp_dom != 0]])

#             print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Train on synthetic data and evaluate on Dp data
#             print('Training on synthetic data...')
#             acc, loss, f1 = train_and_test(x_syn_dom, y_syn_dom, x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs, augment=True)
#             save_scores(src_class, domain, acc, loss, f1, 'Syn_aug', dataset)
#             accs.append(acc)
#             f1s.append(f1)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f} | F1: {f1:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
#     print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")




# def compute_TSTR_Syn_all(dataset):
#     accs = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         # Load synthetic data
#         x_syn, y_syn, k_syn = get_syn_data(dataset, syn_name, src_class)
#         print(f'x_syn.shape: {x_syn.shape} | np.unique(y_syn): {np.unique(y_syn)}')

#         # Load Dp data
#         x_dp, y_dp, k_dp = get_data(dataset, 'dp', src_class)
#         print(f'x_dp.shape: {x_dp.shape} | np.unique(y_dp): {np.unique(y_dp)}\n')

#         # Train on synthetic data and evaluate on Dp data
#         print('Training on synthetic data...')
#         acc, loss = train_and_test(x_syn, y_syn, x_dp, y_dp, dataset, num_epochs=num_epochs)
#         save_scores(src_class, 100, acc, loss, 'Syn_all', dataset)
#         accs.append(acc)
#         print(f'Source class: {src_class} | Domain: All | Accuracy: {acc:.4f} | Loss: {loss:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f}\n")




# def compute_TSTR_Df_Syn(dataset):
#     accs = []
#     f1s = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         # Load Df data
#         x_df, y_df, k_df = get_data(dataset, 'df', src_class)
#         print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)}\n')

#         # Train on Df data
#         print('Training on Df data...')
#         df_model = train_only(x_df, y_df, dataset, num_epochs=num_epochs)

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             # Load synthetic data
#             x_syn_dom, y_syn_dom, k_syn_dom = get_syn_data(dataset, syn_name, src_class, domain)
#             print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             y0_indices_syn = np.where(y_syn_dom == 0)[0]
#             y0_indices_dp = np.where(y_dp_dom == 0)[0]
#             assert np.array_equal(y0_indices_syn, y0_indices_dp), "Labels do not match"
#             assert np.array_equal(y_syn_dom[y0_indices_syn], y_dp_dom[y0_indices_dp]), "Labels do not match"

#             y0_indices_train, y0_indices_test = train_test_split(y0_indices_syn, test_size=0.5, shuffle=True, random_state=seed)

#             x_syn_dom_y0, y_syn_dom_y0 = x_syn_dom[y0_indices_train], y_syn_dom[y0_indices_train]
#             x_dp_dom_y0, y_dp_dom_y0 = x_dp_dom[y0_indices_test], y_dp_dom[y0_indices_test]

#             x_syn_dom = np.concatenate([x_syn_dom_y0, x_syn_dom[y_syn_dom != 0]])
#             y_syn_dom = np.concatenate([y_syn_dom_y0, y_syn_dom[y_syn_dom != 0]])
#             x_dp_dom = np.concatenate([x_dp_dom_y0, x_dp_dom[y_dp_dom != 0]])
#             y_dp_dom = np.concatenate([y_dp_dom_y0, y_dp_dom[y_dp_dom != 0]])

#             print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Fine-tune Df model on synthetic data and evaluate on Dp data
#             print('Fine-tuning on synthetic data...')
#             df_syn_model = fine_tune(copy.deepcopy(df_model), x_syn_dom, y_syn_dom, num_epochs=num_epochs)
#             acc, loss, f1 = evaluate_model(df_syn_model, get_dataloader(x_dp_dom, y_dp_dom))
#             save_scores(src_class, domain, acc, loss, f1, 'Df_Syn', dataset)
#             accs.append(acc)
#             f1s.append(f1)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f} | F1: {f1:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
#     print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")




# def compute_TSTR_Df_Syn_all(dataset):
#     accs = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         # Load Df data
#         x_df, y_df, k_df = get_data(dataset, 'df', src_class)
#         print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)}\n')

#         # Train on Df data
#         print('Training on Df data...')
#         df_model = train_only(x_df, y_df, dataset, num_epochs=num_epochs)

#         # Load synthetic data
#         x_syn, y_syn, k_syn = get_syn_data(dataset, syn_name, src_class)
#         print(f'x_syn.shape: {x_syn.shape} | np.unique(y_syn): {np.unique(y_syn)}')

#         # Load Dp data
#         x_dp, y_dp, k_dp = get_data(dataset, 'dp', src_class)
#         print(f'x_dp.shape: {x_dp.shape} | np.unique(y_dp): {np.unique(y_dp)}\n')

#         # Fine-tune Df model on synthetic data and evaluate on Dp data
#         print('Fine-tuning on synthetic data...')
#         df_syn_model = fine_tune(copy.deepcopy(df_model), x_syn, y_syn, num_epochs=num_epochs)
#         acc, loss = evaluate_model(df_syn_model, get_dataloader(x_dp, y_dp))
#         save_scores(src_class, 100, acc, loss, 'Df_Syn_all', dataset)
#         accs.append(acc)
#         print(f'Source class: {src_class} | Domain: All | Accuracy: {acc:.4f} | Loss: {loss:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f}\n")




# def compute_TSTR_Df_plus_Syn(dataset):
#     accs = []
#     f1s = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         # Load Df data
#         x_df, y_df, k_df = get_data(dataset, 'df', src_class)
#         print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)}\n')

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             # Load synthetic data
#             x_syn_dom, y_syn_dom, k_syn_dom = get_syn_data(dataset, syn_name, src_class, domain)
#             print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             y0_indices_syn = np.where(y_syn_dom == 0)[0]
#             y0_indices_dp = np.where(y_dp_dom == 0)[0]
#             assert np.array_equal(y0_indices_syn, y0_indices_dp), "Labels do not match"
#             assert np.array_equal(y_syn_dom[y0_indices_syn], y_dp_dom[y0_indices_dp]), "Labels do not match"

#             y0_indices_train, y0_indices_test = train_test_split(y0_indices_syn, test_size=0.5, shuffle=True, random_state=seed)

#             x_syn_dom_y0, y_syn_dom_y0 = x_syn_dom[y0_indices_train], y_syn_dom[y0_indices_train]
#             x_dp_dom_y0, y_dp_dom_y0 = x_dp_dom[y0_indices_test], y_dp_dom[y0_indices_test]

#             x_syn_dom = np.concatenate([x_syn_dom_y0, x_syn_dom[y_syn_dom != 0]])
#             y_syn_dom = np.concatenate([y_syn_dom_y0, y_syn_dom[y_syn_dom != 0]])
#             x_dp_dom = np.concatenate([x_dp_dom_y0, x_dp_dom[y_dp_dom != 0]])
#             y_dp_dom = np.concatenate([y_dp_dom_y0, y_dp_dom[y_dp_dom != 0]])

#             print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Join Df and synthetic data
#             x_df_syn = np.concatenate([x_df, x_syn_dom], axis=0)
#             y_df_syn = np.concatenate([y_df, y_syn_dom], axis=0)
#             k_df_syn = np.concatenate([k_df, k_syn_dom], axis=0)
#             print(f'x_df_syn.shape: {x_df_syn.shape} | np.unique(y_df_syn): {np.unique(y_df_syn)}')

#             # Train on Df plus synthetic data and test on Dp data
#             print('Training on Df plus synthetic data...')
#             acc, loss, f1 = train_and_test(x_df, y_df, x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs)
#             save_scores(src_class, domain, acc, loss, f1, 'Df_plus_Syn', dataset)
#             accs.append(acc)
#             f1s.append(f1)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f} | F1: {f1:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
#     print(f"Mean F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}\n")




# def compute_TSTRFS_Dpfs(dataset):
#     accs = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             # Load Dpfs data
#             x_dpfs_dom, y_dpfs_dom, k_dpfs_dom = get_data(dataset, 'dpfs', src_class, domain)
#             print(f'x_dpfs_dom.shape: {x_dpfs_dom.shape} | np.unique(y_dpfs_dom): {np.unique(y_dpfs_dom)}')

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Train on Dpfs data and test on Dp data
#             print('Training on Dpfs data...')
#             acc, loss = train_and_test(x_dpfs_dom, y_dpfs_dom, x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs)
#             save_scores(src_class, domain, acc, loss, 'FS_Dpfs', dataset)
#             accs.append(acc)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f}\n")



# def compute_TSTRFS_Df_Dpfs(dataset):
#     accs = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         # Load Df data
#         x_df, y_df, k_df = get_data(dataset, 'df', src_class)
#         print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)}\n')

#         # Train on Df data
#         print('Training on Df data...')
#         df_model = train_only(x_df, y_df, dataset, num_epochs=num_epochs)

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             # Load Dpfs data
#             x_dpfs_dom, y_dpfs_dom, k_dpfs_dom = get_data(dataset, 'dpfs', src_class, domain)
#             print(f'x_dpfs_dom.shape: {x_dpfs_dom.shape} | np.unique(y_dpfs_dom): {np.unique(y_dpfs_dom)}')

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Fine-tune Df model on Dpfs data and test on Dp data
#             print('Fine-tuning on Dpfs data...')
#             df_dpfs_model = fine_tune(copy.deepcopy(df_model), x_dpfs_dom, y_dpfs_dom, num_epochs=num_epochs)
#             acc, loss = evaluate_model(df_dpfs_model, get_dataloader(x_dp_dom, y_dp_dom))
#             save_scores(src_class, domain, acc, loss, 'FS_Df_Dpfs', dataset)
#             accs.append(acc)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f}\n")


# def compute_TSTRFS_Syn(dataset):
#     accs = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             # Load synthetic data
#             x_syn_dom, y_syn_dom, k_syn_dom = get_syn_data(dataset, syn_name, src_class, domain)
#             print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Train on synthetic data and evaluate on Dp data
#             print('Training on synthetic data...')
#             acc, loss = train_and_test(x_syn_dom, y_syn_dom, x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs)
#             save_scores(src_class, domain, acc, loss, 'FS_Syn', dataset)
#             accs.append(acc)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f}\n")



# def compute_TSTRFS_Df_Syn(dataset):
#     accs = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         # Load Df data
#         x_df, y_df, k_df = get_data(dataset, 'df', src_class)
#         print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)}\n')

#         # Train on Df data
#         print('Training on Df data...')
#         df_model = train_only(x_df, y_df, dataset, num_epochs=num_epochs)

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             # Load synthetic data
#             x_syn_dom, y_syn_dom, k_syn_dom = get_syn_data(dataset, syn_name, src_class, domain)
#             print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Fine-tune Df model on synthetic data and evaluate on Dp data
#             print('Fine-tuning on synthetic data...')
#             df_syn_model = fine_tune(copy.deepcopy(df_model), x_syn_dom, y_syn_dom, num_epochs=num_epochs)
#             acc, loss = evaluate_model(df_syn_model, get_dataloader(x_dp_dom, y_dp_dom))
#             save_scores(src_class, domain, acc, loss, 'FS_Df_Syn', dataset)
#             accs.append(acc)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f}\n")



# def compute_TSTRFS_Syn_Dpfs(dataset):
#     accs = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             # Load synthetic data
#             x_syn_dom, y_syn_dom, k_syn_dom = get_syn_data(dataset, syn_name, src_class, domain)
#             print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')

#             # Load Dpfs data
#             x_dpfs_dom, y_dpfs_dom, k_dpfs_dom = get_data(dataset, 'dpfs', src_class, domain)
#             print(f'x_dpfs_dom.shape: {x_dpfs_dom.shape} | np.unique(y_dpfs_dom): {np.unique(y_dpfs_dom)}')

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Train on synthetic data
#             print('Training on synthetic data...')
#             syn_model = train_only(x_syn_dom, y_syn_dom, dataset, num_epochs=num_epochs)

#             # Fine-tune synthetic model on Dpfs data and evaluate on Dp data
#             print('Fine-tuning on Dpfs data...')
#             syn_dpfs_model = fine_tune(copy.deepcopy(syn_model), x_dpfs_dom, y_dpfs_dom, num_epochs=num_epochs)
#             acc, loss = evaluate_model(syn_dpfs_model, get_dataloader(x_dp_dom, y_dp_dom))
#             save_scores(src_class, domain, acc, loss, 'FS_Syn_Dpfs', dataset)
#             accs.append(acc)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f}\n")



# def compute_TSTRFS_Df_Syn_Dpfs(dataset):
#     accs = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         # Load Df data
#         x_df, y_df, k_df = get_data(dataset, 'df', src_class)
#         print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)}\n')

#         # Train on Df data
#         print('Training on Df data...')
#         df_model = train_only(x_df, y_df, dataset, num_epochs=num_epochs)

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             # Load synthetic data
#             x_syn_dom, y_syn_dom, k_syn_dom = get_syn_data(dataset, syn_name, src_class, domain)
#             print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')

#             # Load Dpfs data
#             x_dpfs_dom, y_dpfs_dom, k_dpfs_dom = get_data(dataset, 'dpfs', src_class, domain)
#             print(f'x_dpfs_dom.shape: {x_dpfs_dom.shape} | np.unique(y_dpfs_dom): {np.unique(y_dpfs_dom)}')

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Fine-tune Df model on synthetic data
#             print('Fine-tuning on synthetic data...')
#             df_syn_model = fine_tune(copy.deepcopy(df_model), x_syn_dom, y_syn_dom, num_epochs=num_epochs)

#             # Fine-tune Df-syn model on Dpfs data and evaluate on Dp data
#             print('Fine-tuning on Dpfs data...')
#             df_syn_dpfs_model = fine_tune(copy.deepcopy(df_syn_model), x_dpfs_dom, y_dpfs_dom, num_epochs=num_epochs)
#             acc, loss = evaluate_model(df_syn_dpfs_model, get_dataloader(x_dp_dom, y_dp_dom))
#             save_scores(src_class, domain, acc, loss, 'FS_Df_Syn_Dpfs', dataset)
#             accs.append(acc)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f}\n")



# def compute_TSTRFS_Df_plus_Dpfs(dataset):
#     accs = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         # Load Df data
#         x_df, y_df, k_df = get_data(dataset, 'df', src_class)
#         print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)}\n')

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             # Load Dpfs data
#             x_dpfs_dom, y_dpfs_dom, k_dpfs_dom = get_data(dataset, 'dpfs', src_class, domain)
#             print(f'x_dpfs_dom.shape: {x_dpfs_dom.shape} | np.unique(y_dpfs_dom): {np.unique(y_dpfs_dom)}')

#             # Join Df and Dpfs data
#             x_df_dpfs = np.concatenate([x_df, x_dpfs_dom], axis=0)
#             y_df_dpfs = np.concatenate([y_df, y_dpfs_dom], axis=0)
#             k_df_dpfs = np.concatenate([k_df, k_dpfs_dom], axis=0)
#             print(f'x_df_dpfs.shape: {x_df_dpfs.shape} | np.unique(y_df_dpfs): {np.unique(y_df_dpfs)}')

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Train on Df plus Dpfs data and test on Dp data
#             print('Training on Df plus Dpfs data...')
#             acc, loss = train_and_test(x_df_dpfs, y_df_dpfs, x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs)
#             save_scores(src_class, domain, acc, loss, 'FS_Df_plus_Dpfs', dataset)
#             accs.append(acc)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f}\n")


# def compute_TSTRFS_Df_plus_Syn(dataset):
#     accs = []

#     for src_class in config[dataset]['class_names']:
#         if src_class != 'WAL':
#             continue

#         print(f"Source class: {src_class}\n")

#         # Load Df data
#         x_df, y_df, k_df = get_data(dataset, 'df', src_class)
#         print(f'x_df.shape: {x_df.shape} | np.unique(y_df): {np.unique(y_df)}\n')

#         for domain in range(config[dataset]['num_df_domains'], config[dataset]['num_df_domains'] + config[dataset]['num_dp_domains']):
#             print(f"Domain: {domain}")

#             # Load synthetic data
#             x_syn_dom, y_syn_dom, k_syn_dom = get_syn_data(dataset, syn_name, src_class, domain)
#             print(f'x_syn_dom.shape: {x_syn_dom.shape} | np.unique(y_syn_dom): {np.unique(y_syn_dom)}')

#             # Join Df and synthetic data
#             x_df_syn = np.concatenate([x_df, x_syn_dom], axis=0)
#             y_df_syn = np.concatenate([y_df, y_syn_dom], axis=0)
#             k_df_syn = np.concatenate([k_df, k_syn_dom], axis=0)
#             print(f'x_df_syn.shape: {x_df_syn.shape} | np.unique(y_df_syn): {np.unique(y_df_syn)}')

#             # Load Dp data
#             x_dp_dom, y_dp_dom, k_dp_dom = get_data(dataset, 'dp', src_class, domain)
#             print(f'x_dp_dom.shape: {x_dp_dom.shape} | np.unique(y_dp_dom): {np.unique(y_dp_dom)}\n')

#             # Train on Df plus synthetic data and test on Dp data
#             print('Training on Df plus synthetic data...')
#             acc, loss = train_and_test(x_df, y_df, x_dp_dom, y_dp_dom, dataset, num_epochs=num_epochs)
#             save_scores(src_class, domain, acc, loss, 'FS_Df_plus_Syn', dataset)
#             accs.append(acc)
#             print(f'Source class: {src_class} | Domain: {domain} | Accuracy: {acc:.4f} | Loss: {loss:.4f}\n')

#     print(f"Mean accuracy: {np.mean(accs):.4f}\n")



