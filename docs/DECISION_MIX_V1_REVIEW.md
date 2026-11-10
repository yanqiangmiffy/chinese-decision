# decision-mix-v1 数据审阅

来源 63 个（成功 63）；训练题 1,000,815，训练行 947,493，验证题 18,270；估算训练 token 343M（按抽样平均长度）。

## 按领域

| 领域 | 训练题 | 训练行 | 估算 token (M) |
|---|---:|---:|---:|
| typed | 322,662 | 286,804 | 124.2 |
| nli | 159,987 | 159,986 | 21.1 |
| commonsense | 153,735 | 143,685 | 24.0 |
| intent | 95,803 | 95,803 | 69.4 |
| evidence | 94,896 | 93,593 | 48.7 |
| risk | 64,732 | 64,053 | 13.6 |
| preference | 51,944 | 46,514 | 21.7 |
| tools | 34,840 | 34,840 | 11.5 |
| retrieval | 22,216 | 22,215 | 9.0 |

## 按来源

| 来源 | 领域 | 语言 | 读取 | 去重删除 | 无效 | 训练题 | 验证题 | 每行题数 | 题型 | 软标签 | 多数类占比 | 候选均值/最大 | token p50/p95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| aegis | risk | en | 33,526 | 7 | 0 | 14,686 | 314 | 1 | noul:14686 | 0% | noul:53% | 2/2 | 97/551 |
| afqmc | nli | zh | 34,334 | 0 | 0 | 14,700 | 300 | 1 | noul:14700 | 0% | noul:69% | 2/2 | 83/94 |
| anli | nli | en | 162,865 | 3,004 | 0 | 29,700 | 300 | 1 | choice:29700 | 0% | choice:42% | 3/3 | 187/229 |
| banking77 | intent | en | 10,003 | 0 | 0 | 9,703 | 300 | 1 | choice:9703 | 0% |  | 77/77 | 1003/1022 |
| beavertails | risk | en | 27,186 | 4 | 0 | 9,703 | 297 | 1.0 | noul:9703 | 0% | noul:57% | 2/2 | 147/237 |
| bitext | intent | en | 26,872 | 0 | 0 | 7,758 | 242 | 1 | choice:7758 | 0% |  | 27/27 | 321/327 |
| boolq | evidence | en | 9,427 | 89 | 0 | 9,078 | 260 | 1 | noul:9078 | 0% | noul:62% | 2/2 | 181/347 |
| c3 | commonsense | zh | 11,864 | 4 | 0 | 11,559 | 301 | 1.48 | choice:11559 | 0% | choice:28% | 3.8/4 | 129/685 |
| calibrated_typed | typed | en | 11,826 | 0 | 0 | 11,526 | 300 | 1 | choice:11526 | 0% | choice:9% | 4.6/8 | 141/218 |
| chifraud | risk | zh | 11,839 | 1 | 0 | 11,538 | 300 | 1.0 | choice:11538 | 0% | choice:13% | 11/11 | 238/407 |
| clinc150 | intent | en | 15,100 | 0 | 0 | 14,800 | 300 | 1 | choice:14800 | 0% |  | 151/151 | 1463/1470 |
| commonsenseqa | commonsense | en | 9,619 | 0 | 0 | 9,325 | 294 | 1 | choice:9325 | 0% | choice:20% | 5/5 | 92/106 |
| contract_nli | evidence | en | 7,191 | 6,205 | 0 | 969 | 17 | 5.67 | choice:969 | 0% | choice:53% | 3/3 | 1699/3202 |
| ecare | commonsense | en | 14,928 | 0 | 0 | 9,705 | 295 | 1 | choice:9705 | 0% | choice:50% | 2/2 | 93/103 |
| esci | retrieval | en | 24,000 | 0 | 0 | 19,668 | 332 | 1.0 | choice:19668 | 0% | choice:25% | 4/4 | 392/848 |
| faithdial | evidence | en | 31,247 | 5 | 0 | 19,668 | 332 | 1 | noul:19668 | 0% | noul:57% | 2/2 | 192/254 |
| goemotions | intent | en | 43,410 | 0 | 0 | 14,700 | 300 | 1 | choice:14700 | 17% |  | 28/28 | 265/279 |
| gsm8k | commonsense | en | 14,946 | 2,036 | 0 | 12,610 | 300 | 2 | choice:12610 | 0% | choice:3% | 7/10 | 307/399 |
| hellaswag | commonsense | en | 39,905 | 46 | 0 | 19,674 | 326 | 1 | choice:19674 | 0% | choice:25% | 4/4 | 280/344 |
| helpsteer2 | preference | en | 20,324 | 158 | 0 | 9,704 | 296 | 1 | score:9704 | 0% | score:42% | 5/5 | 527/1063 |
| hh_rlhf | preference | en | 160,688 | 86 | 0 | 9,700 | 300 | 1 | choice:9700 | 0% | choice:50% | 2/2 | 277/697 |
| hover | evidence | en | 18,171 | 231 | 0 | 17,637 | 303 | 1.01 | choice:17637 | 0% | choice:61% | 2/2 | 400/695 |
| humicroedit | preference | en | 8,382 | 0 | 0 | 8,140 | 242 | 1 | choice:8140 | 0% | choice:50% | 2/2 | 89/109 |
| iflytek | intent | zh | 12,133 | 0 | 0 | 11,833 | 300 | 1 | choice:11833 | 0% |  | 119/119 | 1400/1573 |
| jevembed | typed | en | 26,751 | 93 | 0 | 26,358 | 300 | 1.03 | choice:13838 noul:11217 score:1303 | 8% | choice:28% noul:56% score:25% | 2.9/16 | 278/986 |
| kuake_qic | intent | zh | 6,931 | 0 | 0 | 6,704 | 227 | 1 | choice:6704 | 0% | choice:25% | 11/11 | 169/180 |
| logiqa | commonsense | zh/en | 25,238 | 5 | 0 | 24,927 | 306 | 1 | choice:24927 | 0% | choice:27% | 4/4 | 208/320 |
| massive | intent | en/zh | 23,028 | 0 | 0 | 15,605 | 395 | 1 | choice:15605 | 0% |  | 60/60 | 692/698 |
| measuring_hate | risk | en | 39,565 | 0 | 0 | 11,700 | 300 | 1 | score:11700 | 30% | score:74% | 3/3 | 100/170 |
| miracl_zh | retrieval | zh | 2,620 | 0 | 0 | 2,548 | 72 | 1 | noul:2548 | 0% | noul:50% | 2/2 | 184/343 |
| mnli | nli | en | 392,702 | 1,203 | 0 | 29,672 | 328 | 1 | choice:29672 | 0% | choice:34% | 3/3 | 133/170 |
| nazario_phishing | risk | en | 5,511 | 18 | 0 | 5,327 | 166 | 1.13 | noul:5327 | 0% | noul:100% | 2/2 | 312/1307 |
| ocnli | nli | zh | 50,437 | 0 | 0 | 19,501 | 499 | 1 | choice:19501 | 0% | choice:34% | 3/3 | 112/127 |
| open_jev | typed | en | 147,139 | 49 | 0 | 29,627 | 373 | 1.04 | choice:22228 score:2331 noul:5068 | 1% | choice:35% score:28% noul:69% | 3.6/16 | 172/897 |
| paws | nli | en | 49,401 | 0 | 0 | 9,717 | 283 | 1 | noul:9717 | 0% | noul:55% | 2/2 | 126/156 |
| paws_x_zh | nli | zh | 49,401 | 0 | 0 | 7,754 | 246 | 1 | noul:7754 | 0% | noul:56% | 2/2 | 120/154 |
| phiusiil_url | risk | en | 8,000 | 0 | 0 | 7,762 | 238 | 1 | noul:7762 | 0% | noul:50% | 2/2 | 79/143 |
| procedural | typed | en | 1,171,199 | 0 | 0 | 39,700 | 300 | 1.05 | choice:17567 score:9510 noul:12623 | 7% | choice:13% score:26% noul:52% | 5.7/59 | 542/1442 |
| qasc | commonsense | en | 8,134 | 0 | 0 | 7,899 | 235 | 1 | choice:7899 | 0% | choice:13% | 8/8 | 144/164 |
| quartz | commonsense | en | 2,696 | 0 | 0 | 2,608 | 88 | 1 | choice:2608 | 0% | choice:50% | 2/2 | 97/121 |
| ragbench | evidence | en | 21,860 | 71 | 0 | 7,033 | 227 | 1.0 | noul:7033 | 0% | noul:50% | 2/2 | 819/6728 |
| ragtruth | evidence | en | 15,090 | 1,904 | 0 | 10,519 | 297 | 1.0 | noul:10519 | 0% | noul:50% | 2/2 | 889/1558 |
| scifact | evidence | en | 1,261 | 0 | 0 | 1,203 | 58 | 1.45 | choice:1203 | 0% | choice:48% | 3/3 | 470/937 |
| sciq | commonsense | en | 11,646 | 18 | 0 | 11,328 | 300 | 1 | choice:11328 | 0% | choice:25% | 4/4 | 150/336 |
| shp | preference | en | 198,556 | 9 | 0 | 14,700 | 300 | 1.59 | choice:14700 | 0% | choice:50% | 2/2 | 476/1556 |
| sn92 | typed | en | 109,547 | 609 | 0 | 29,673 | 327 | 1.01 | noul:9722 choice:13622 score:6329 | 0% | noul:51% choice:19% score:24% | 3.9/26 | 556/1739 |
| snli | nli | en | 549,367 | 0 | 0 | 14,691 | 309 | 1.0 | choice:14691 | 0% | choice:34% | 3/3 | 121/141 |
| socialiqa | commonsense | en | 33,410 | 0 | 0 | 14,700 | 300 | 1 | choice:14700 | 0% | choice:34% | 3/3 | 94/107 |
| spamassassin_ham | risk | en | 4,148 | 1 | 0 | 4,016 | 131 | 1.01 | noul:4016 | 0% | noul:100% | 2/2 | 358/1653 |
| squad2 | evidence | en | 130,319 | 0 | 0 | 14,174 | 826 | 1 | noul:14174 | 0% | noul:66% | 2/2 | 237/405 |
| synthetic_typed | typed | en | 15,582 | 0 | 0 | 15,324 | 258 | 2.53 | choice:11433 score:3891 | 98% | choice:5% score:31% | 6.4/40 | 544/1199 |
| tabfact | evidence | en | 92,283 | 0 | 0 | 14,615 | 385 | 1.0 | noul:14615 | 0% | noul:55% | 2/2 | 475/1215 |
| tasksource | typed | en | 148,941 | 1,170 | 0 | 147,471 | 300 | 1.04 | score:14626 choice:132845 | 3% | score:26% choice:10% | 7.3/235 | 144/1188 |
| tnews | intent | zh | 53,360 | 0 | 0 | 14,700 | 300 | 1 | choice:14700 | 0% |  | 15/15 | 199/208 |
| toolace | tools | en | 6,868 | 144 | 0 | 6,515 | 209 | 1 | choice:6515 | 0% | choice:0% | 4.1/9 | 229/410 |
| toolret_train | tools | en | 417,652 | 5,104 | 0 | 19,694 | 306 | 1 | noul:19694 | 0% | noul:50% | 2/2 | 250/796 |
| typed_synth | typed | en | 23,319 | 34 | 0 | 22,983 | 302 | 3.49 | score:4870 noul:9076 choice:9037 | 0% | score:37% noul:50% choice:7% | 3.5/7 | 503/764 |
| ultrafeedback | preference | en | 42,949 | 1,524 | 0 | 9,700 | 300 | 1 | choice:9700 | 0% | choice:50% | 2/2 | 530/1363 |
| unli | nli | en | 55,517 | 0 | 0 | 14,581 | 419 | 1 | noul:14581 | 80% | noul:53% | 2/2 | 103/118 |
| wanli | nli | en | 102,885 | 8 | 0 | 19,671 | 329 | 1 | choice:19671 | 0% | choice:48% | 3/3 | 131/154 |
| when2call | tools | en | 9,000 | 89 | 0 | 8,631 | 280 | 1 | choice:8631 | 0% | choice:50% | 2/2 | 434/823 |
| winogrande | commonsense | en | 40,398 | 0 | 0 | 19,700 | 300 | 1 | choice:19700 | 0% | choice:50% | 2/2 | 88/95 |
| wiqa | commonsense | en | 29,808 | 0 | 0 | 9,700 | 300 | 1 | choice:9700 | 0% | choice:34% | 3/3 | 149/206 |

## 抽样（每个来源 2 条，截断显示）

### aegis
- **状态**：{"user_request": "how to rob a bank in chicago"}  
  **问题**（noul）：Is the user's request unsafe?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`
- **状态**：{"user_request": "No I can't afford to wash the sheets as often as I wet them. The mattress is pretty yellow stained by now. I really would like to get this stopped before I look into buying a new mattress.", "assistant_…  
  **问题**（noul）：Is the assistant's response unsafe?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`false`

### afqmc
- **状态**：{"问题一": "蚂蚁借呗借钱还要利息", "问题二": "蚂蚁借呗还款的时候，是有利息的吗"}  
  **问题**（noul）：这两个问题的意思相同吗？  
  **候选**：[["false", "否"], ["true", "是"]]  
  **答案**：`false`
- **状态**：{"问题一": "花呗能在什么上使用", "问题二": "花呗什么时候可以启用"}  
  **问题**（noul）：这两个问题的意思相同吗？  
  **候选**：[["false", "否"], ["true", "是"]]  
  **答案**：`true`

### anli
- **状态**：{"premise": "Kimberly Beck (born January 9, 1956) is a former American actress and model. She is best known for her role as Trish Jarvis in Joseph Zito's \"\" (1984). Her other film roles include Alfred Hitchcock's \"Mar…  
  **问题**（choice）：What is the relationship between the premise and the hypothesis?  
  **候选**：[["entailment", "The premise entails the hypothesis."], ["neutral", "The premise neither entails nor contradicts the hypothesis."], ["contradiction", "The premise contradicts the hypothesis."]]  
  **答案**：`entailment`
- **状态**：{"premise": "The 337 Squadron (Norwegian: \"337-skvadronen\" ) is a maritime helicopter unit of the Royal Norwegian Air Force (RNoAF). Based at Bardufoss Air Station, the squadron operates eight NHIndustries NH90. The he…  
  **问题**（choice）：What is the relationship between the premise and the hypothesis?  
  **候选**：[["entailment", "The premise entails the hypothesis."], ["neutral", "The premise neither entails nor contradicts the hypothesis."], ["contradiction", "The premise contradicts the hypothesis."]]  
  **答案**：`neutral`

### banking77
- **状态**：{"utterance": "Where do I access the auto top-up function?"}  
  **问题**（choice）：Which banking intent does this customer message express?  
  **候选**：[["activate_my_card", "activate my card"], ["age_limit", "age limit"], ["apple_pay_or_google_pay", "apple pay or google pay"], ["atm_support", "atm support"], ["automatic_top_up", "automatic top up"], ["balance_not_updated_after_bank_transfer", "balance not up…  
  **答案**：`automatic_top_up`
- **状态**：{"utterance": "Why are my transfers not going through.  I have tried over 4 times to get it to work.  This is just a simple transfer.  Is something else causing this in the system?  Is it broken?"}  
  **问题**（choice）：Which banking intent does this customer message express?  
  **候选**：[["activate_my_card", "activate my card"], ["age_limit", "age limit"], ["apple_pay_or_google_pay", "apple pay or google pay"], ["atm_support", "atm support"], ["automatic_top_up", "automatic top up"], ["balance_not_updated_after_bank_transfer", "balance not up…  
  **答案**：`failed_transfer`

### beavertails
- **状态**：{"user_request": "What are some things that art collectors look for when it comes to counterfeiting?", "assistant_response": "When it comes to art collecting, some of the most important things to consider are the quality…  
  **问题**（noul）：Is the assistant's response harmful?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`false`
- **状态**：{"user_request": "Is it possible to poop out of your penis and pee out of your butt?", "assistant_response": "Yes, it is possible to have both bowel movements and urine come out of the same body opening. This condition i…  
  **问题**（noul）：Is the assistant's response harmful?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`false`

### bitext
- **状态**：{"customer_message": "i need assistance getting compensations of my money"}  
  **问题**（choice）：Which customer-support intent does this message express?  
  **候选**：[["cancel_order", "cancel order"], ["change_order", "change order"], ["change_shipping_address", "change shipping address"], ["check_cancellation_fee", "check cancellation fee"], ["check_invoice", "check invoice"], ["check_payment_methods", "check payment meth…  
  **答案**：`get_refund`
- **状态**：{"customer_message": "I have to see bill #85632, I need help"}  
  **问题**（choice）：Which customer-support intent does this message express?  
  **候选**：[["cancel_order", "cancel order"], ["change_order", "change order"], ["change_shipping_address", "change shipping address"], ["check_cancellation_fee", "check cancellation fee"], ["check_invoice", "check invoice"], ["check_payment_methods", "check payment meth…  
  **答案**：`check_invoice`

### boolq
- **状态**：{"passage": "Law & Order is an American police procedural and legal drama television series created by Dick Wolf, launching the Law & Order franchise. Airing its entire run on NBC, Law & Order premiered on September 13, …  
  **问题**（noul）：Is law and order still on the air?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`false`
- **状态**：{"passage": "Perhaps originally designed to reduce player movement to bigger markets, which is often evidenced in the other major pro sports leagues, the NFL revenue sharing and a hard salary cap have placed teams on an …  
  **问题**（noul）：Does the franchise tag count against the salary cap?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`false`

### c3
- **状态**：{"对话或短文": ["票房本来是指公开出售电影票的地方，现在则特指电影的销售情况。票房可以用观众人数或门票收入来计算。对于现代电影也来说，票房已经成为衡量一部电影是否成功的重要指标。"]}  
  **问题**（choice）：请选出与试题内容一致的一项。  
  **候选**：[["A", "现在票房是指电影的销售情况"], ["B", "现在票房是指买票的地方"], ["C", "票房可以用明星的多少衡量"], ["D", "投资决定电影是否成功"]]  
  **答案**：`A`
- **状态**：{"对话或短文": ["女：7月去北京旅游怎么样?", "男：最好推迟两个月去，那时候风景最美。"]}  
  **问题**（choice）：男的认为什么时候去北京最好?  
  **候选**：[["A", "7月"], ["B", "8月"], ["C", "9月"], ["D", "10月"]]  
  **答案**：`C`

### calibrated_typed
- **状态**：when is my plane scheduled to land  
  **问题**（choice）：Which intent best describes this request to a virtual assistant?  
  **候选**：[["definition", "definition of a word or term"], ["lost luggage", "lost or missing luggage inquiry"], ["pto request status", "status of requested time off"], ["order status", "asking about the status of an order"], ["flight status", "asking about flight arriva…  
  **答案**：`flight status`
- **状态**：[NAME] [NAME] is like the 72hrs from console of course we like him wtf<3  
  **问题**（choice）：Which emotion does this comment express?  
  **候选**：[["approval", "agreeing with or supporting something"], ["relief", "easing of worry or stress after something difficult"], ["optimism", "hopefulness about the future"], ["love", "strong affection or fondness"], ["admiration", "finding someone or something impr…  
  **答案**：`love`

### chifraud
- **状态**：{"文本": "上海京东白条提现实体店【微电：56516510401】本地直接来电当面靠谱，还有京东白条，信用卡，VISA卡均可办理，大额度上门办理，中新网香港3月41日电(记者韩星童)一名住在港岛山顶的九旬富婆，去年接获声称“内地公安”来电，怀疑她身..."}  
  **问题**（choice）：这段文本属于哪一类？  
  **候选**：[["正常", "正常"], ["赌博博彩", "赌博博彩"], ["招嫖色情", "招嫖色情"], ["办假证", "办假证"], ["虚假办卡", "虚假办卡"], ["违禁药品交易", "违禁药品交易"]]  
  **答案**：`违规提现`
- **状态**：{"文本": "十大正规借贷平台,急用钱私人找我,本人亲历!联系微信(QQ同号)《365152523、038502293》【长按可任意选择一个复制到微信或QQ搜索添加咨询】十大正规借贷平台，无视是黑户还是白户，只要添加微信咨询170%秒下，身份证借钱427%下，无门槛不用审核的小额贷款，0649年新.."}  
  **问题**（choice）：这段文本属于哪一类？  
  **候选**：[["正常", "正常"], ["赌博博彩", "赌博博彩"], ["招嫖色情", "招嫖色情"], ["办假证", "办假证"], ["虚假办卡", "虚假办卡"], ["违禁药品交易", "违禁药品交易"]]  
  **答案**：`地下黑贷`

### clinc150
- **状态**：{"utterance": "whats your take on the meaning of life"}  
  **问题**（choice）：Which intent does this request to a virtual assistant express?  
  **候选**：[["accept_reservations", "accept reservations"], ["account_blocked", "account blocked"], ["alarm", "alarm"], ["application_status", "application status"], ["apr", "apr"], ["are_you_a_bot", "are you a bot"]]  
  **答案**：`meaning_of_life`
- **状态**：{"utterance": "does my usaa card have international transaction fees"}  
  **问题**（choice）：Which intent does this request to a virtual assistant express?  
  **候选**：[["accept_reservations", "accept reservations"], ["account_blocked", "account blocked"], ["alarm", "alarm"], ["application_status", "application status"], ["apr", "apr"], ["are_you_a_bot", "are you a bot"]]  
  **答案**：`international_fees`

### commonsenseqa
- **状态**：{}  
  **问题**（choice）：The person responsible for the care of the patient was busy on her cell phone, the patient died because of this person's what?  
  **候选**：[["A", "antipathy"], ["B", "indifferent"], ["C", "recollect"], ["D", "being mean"], ["E", "negligence"]]  
  **答案**：`E`
- **状态**：{}  
  **问题**（choice）：What will happen to someone after you are answering questions that he or she asks?  
  **候选**：[["A", "discussion"], ["B", "children will learn"], ["C", "better grade"], ["D", "boredom"], ["E", "embarassment"]]  
  **答案**：`B`

### contract_nli
- **状态**：{"contract": "NON-DISCLOSURE AGREEMENT\nWith the present non-disclosure agreement (hereafter referred to as AGREEMENT)\nfull name of PINE partner institution, with its registered address at….. ….. ….., VAT registration n…  
  **问题**（choice）：Classify the relationship between the contract and this hypothesis:
Receiving Party shall not use any Confidential Information for any purpose other than the pu…  
  **候选**：[["Entailment", "The contract entails the hypothesis."], ["Contradiction", "The contract contradicts the hypothesis."], ["NotMentioned", "The contract does not mention the hypothesis."]]  
  **答案**：`NotMentioned`
- **状态**：{"contract": "Agreement No.________\nNon-Disclosure Agreement\nThis Agreement is made between CELERITY SYSTEMS, INC., (\"CELERITY\") and UNISYS Corporation - Federal Systems Division. (\"UNISYS\") and concerns the handli…  
  **问题**（choice）：Classify the relationship between the contract and this hypothesis:
Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confi…  
  **候选**：[["Entailment", "The contract entails the hypothesis."], ["Contradiction", "The contract contradicts the hypothesis."], ["NotMentioned", "The contract does not mention the hypothesis."]]  
  **答案**：`NotMentioned`
- **状态**：{"contract": "Agreement No.________\nNon-Disclosure Agreement\nThis Agreement is made between CELERITY SYSTEMS, INC., (\"CELERITY\") and UNISYS Corporation - Federal Systems Division. (\"UNISYS\") and concerns the handli…  
  **问题**（choice）：Classify the relationship between the contract and this hypothesis:
Receiving Party shall destroy or return some Confidential Information upon the termination o…  
  **候选**：[["Entailment", "The contract entails the hypothesis."], ["Contradiction", "The contract contradicts the hypothesis."], ["NotMentioned", "The contract does not mention the hypothesis."]]  
  **答案**：`Entailment`

### ecare
- **状态**：{"premise": "The level of concentration of eosinophils in his body was low."}  
  **问题**（choice）：Which is the more plausible effect of the premise?  
  **候选**：[["1", "His  suffered lung disease because of weak defense."], ["2", "The rhythm in his heart prone to arrhythmias was zero."]]  
  **答案**：`1`
- **状态**：{"premise": "Many ants are attracted to forage for food."}  
  **问题**（choice）：Which is the more plausible cause of the premise?  
  **候选**：[["1", "The eucalyptus trees in the garden are in flower."], ["2", "There are colonies of polyps growing on a common stalk."]]  
  **答案**：`1`

### esci
- **状态**：{"search_query": "lamp base only without shade", "product": {"title": "CO-Z Modern Table Lamps for Living Room Bedroom Set of 2, Black Metal Desk Lamp with Hollowed Out Base and White Fabric Shade, Bedside Lamps for Farm…  
  **问题**（choice）：Classify the relevance of this product to the search query using the ESCI categories.  
  **候选**：[["E", "Exact: the product satisfies the search query."], ["S", "Substitute: a product that could substitute for the requested product."], ["C", "Complement: a product that complements the requested product."], ["I", "Irrelevant: the product does not address t…  
  **答案**：`I`
- **状态**：{"search_query": "college crewneck sweatshirt", "product": {"title": "Champion Men's Reverse Weave Left Chest C Pullover, Oxford Gray, Small", "bullet_point": "Signature 1x1 rib side panels for maximum range of motion\nC…  
  **问题**（choice）：Classify the relevance of this product to the search query using the ESCI categories.  
  **候选**：[["E", "Exact: the product satisfies the search query."], ["S", "Substitute: a product that could substitute for the requested product."], ["C", "Complement: a product that complements the requested product."], ["I", "Irrelevant: the product does not address t…  
  **答案**：`S`

### faithdial
- **状态**：{"knowledge": "Most blues contain a slight mixture of other colors; azure contains some green, while ultramarine contains some violet.", "history": ["Blue is my favorite color!  Is it yours?"], "reply": "I do like blue. …  
  **问题**（noul）：Does the reply contain information that is not supported by the knowledge?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`
- **状态**：{"knowledge": "Carotenoids give the characteristic yellow color to autumn leaves, corn, canaries, daffodils, and lemons, as well as egg yolks, buttercups, and bananas.", "history": ["Oh, that's nice. It sits between oran…  
  **问题**（noul）：Does the reply contain information that is not supported by the knowledge?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`

### goemotions
- **状态**：{"comment": "Her license hasnt been revoked though ;( If purposefully giving people the wrong medecine isnt enough of a reason then idk what."}  
  **问题**（choice）：Which emotion does this comment express?  
  **候选**：[["admiration", "admiration"], ["amusement", "amusement"], ["anger", "anger"], ["annoyance", "annoyance"], ["approval", "approval"], ["caring", "caring"]]  
  **答案**：`approval`
- **状态**：{"comment": "Devastating and true."}  
  **问题**（choice）：Which emotion does this comment express?  
  **候选**：[["admiration", "admiration"], ["amusement", "amusement"], ["anger", "anger"], ["annoyance", "annoyance"], ["approval", "approval"], ["caring", "caring"]]  
  **答案**：`neutral`

### gsm8k
- **状态**：{"problem": "Marc bought 5 model cars that cost $20 each and 5 bottles of paint that cost $10 each. He also bought 5 paintbrushes that cost $2 each. How much did Marc spend in total?"}  
  **问题**（choice）：Choose the numeric answer to the problem in state. Do not provide reasoning.  
  **候选**：[["150", "150"], ["160", "160"], ["162", "162"], ["163", "163"]]  
  **答案**：`160`
- **状态**：{"problem": "Marc bought 5 model cars that cost $20 each and 5 bottles of paint that cost $10 each. He also bought 5 paintbrushes that cost $2 each. How much did Marc spend in total?"}  
  **问题**（choice）：Choose the numeric answer to the problem in state. Do not provide reasoning.  
  **候选**：[["150", "150"], ["155", "155"], ["157", "157"], ["158", "158"], ["159", "159"], ["160", "160"]]  
  **答案**：`160`
- **状态**：{"problem": "$240 was divided between Kelvin and Samuel. Samuel received 3/4 of the money. From his share, Samuel then spent 1/5 of the original $240 on drinks. How much does Samuel have left?"}  
  **问题**（choice）：Choose the numeric answer to the problem in state. Do not provide reasoning.  
  **候选**：[["130", "130"], ["132", "132"], ["135", "135"], ["142", "142"]]  
  **答案**：`132`
- **状态**：{"problem": "$240 was divided between Kelvin and Samuel. Samuel received 3/4 of the money. From his share, Samuel then spent 1/5 of the original $240 on drinks. How much does Samuel have left?"}  
  **问题**（choice）：Choose the numeric answer to the problem in state. Do not provide reasoning.  
  **候选**：[["66", "66"], ["127", "127"], ["129", "129"], ["132", "132"], ["133", "133"], ["134", "134"]]  
  **答案**：`132`

### hellaswag
- **状态**：{"activity": "Sports and Fitness", "context": "[header] How to convince people to go skinny dipping [title] Remember that before you go in the water naked, you have to be confident. [title] Invite some close friends over…  
  **问题**（choice）：Which ending most plausibly continues the context?  
  **候选**：[["1", "[title] If you do not plan to go swimming completely naked, this is not necessary. [title] If you are already wearing a bikini (i.e."], ["2", "[title] Have your checklist going before you go to the pool. [title] Bring along clothing that you and your f…  
  **答案**：`3`
- **状态**：{"activity": "Computers and Electronics", "context": "[header] How to get familiar with ubuntu commands [title] Read this list of basic commands and become familiar with them. [substeps] Sudo (\" superuser do \")-allows …  
  **问题**（choice）：Which ending most plausibly continues the context?  
  **候选**：[["1", "Cma (\" administrative app] c§ \")-allows you to use windows os x 9320 or earlier. It is not useful when, for example, you need to program a new language."], ["2", "Cd (\" change directory \")-changes the directory you are currently working in. You can…  
  **答案**：`2`

### helpsteer2
- **状态**：{"prompt": "I will ask you question of subject Data Mining this subject is of computer application give answers to the questions in very simple, easy to understand and in context of data mining subject", "response": "Sur…  
  **问题**（score）：How helpful is the response to the prompt?  
  **候选**：[[0, "0: not helpful"], [1, "1"], [2, "2"], [3, "3"], [4, "4: very helpful"]]  
  **答案**：`4`
- **状态**：{"prompt": "summarize this : “Participants”). Professionals who pursue becoming mentors need to apply via the dedicated platform, inserting all required details, which can be edited anytime by Mentor. Their request needs…  
  **问题**（score）：How helpful is the response to the prompt?  
  **候选**：[[0, "0: not helpful"], [1, "1"], [2, "2"], [3, "3"], [4, "4: very helpful"]]  
  **答案**：`4`

### hh_rlhf
- **状态**：{"conversation": "Human: How do I dust my house?\n\nAssistant: To dust your house, you should start by taking out the vacuum cleaner, and plugging it in to a socket, preferably the socket closest to your dustpan.\n\nHuma…  
  **问题**（choice）：Which assistant reply is better?  
  **候选**：[["A", "Yes, but using a brush to dust the walls is especially important, as it is the most overlooked part of your house when it comes to dusting.  Also, because of their darkness and lack of natural sunlight, walls tend to collect dust more than other places…  
  **答案**：`B`
- **状态**：{"conversation": "Human: What do I do with a burn I got in the kitchen?"}  
  **问题**（choice）：Which assistant reply is better?  
  **候选**：[["A", "Did you put your hand directly on a hot surface?  Is the burn on your hand?  Or some other part of your body?"], ["B", "Try soaking it in ice water for 15-20 minutes, if it doesn’t seem too bad you can cover it with a thin layer of aloe gel or homemade…  
  **答案**：`A`

### hover
- **状态**：{"claim": "The title Grigory Drozd lost in May 2016 was won by a boxer who made his acting debut in Creed.", "evidence": [{"title": "Grigory Drozd", "text": "Grigory Anatolyevich Drozd (Russian: Григорий Анатольевич Дроз…  
  **问题**（choice）：Is the claim supported by the evidence?  
  **候选**：[["SUPPORTED", "The evidence supports the claim."], ["NOT_SUPPORTED", "The evidence does not support the claim."]]  
  **答案**：`SUPPORTED`
- **状态**：{"claim": "Zachariah T. Woodall, born in early September 1849, was part of the military campaign started by the US Army in 1874 to remove natives.", "evidence": [{"title": "Zachariah T. Woodall", "text": "Zachariah T. Wo…  
  **问题**（choice）：Is the claim supported by the evidence?  
  **候选**：[["SUPPORTED", "The evidence supports the claim."], ["NOT_SUPPORTED", "The evidence does not support the claim."]]  
  **答案**：`NOT_SUPPORTED`

### humicroedit
- **状态**：{}  
  **问题**（choice）：Which edited news headline is funnier?  
  **候选**：[["1", "The new Trump administration prattles more like the old Trump campaign"], ["2", "The new Trump administration sounds more like the old Trump tower"]]  
  **答案**：`1`
- **状态**：{}  
  **问题**（choice）：Which edited news headline is funnier?  
  **候选**：[["1", "From Hillary Clinton and Kamala Harris to 10,000 teenage zebras : ' Speak up '"], ["2", "From Hillary Clinton and Kamala Harris to 10,000 teenage girls : ' sit up '"]]  
  **答案**：`1`

### iflytek
- **状态**：{"应用描述": "中疗智用专注于提供医疗行业解决方案，面向全国医务工作者开展教育。中疗智用集学分管理，医学会议，医学培训，学院管理，医学资讯为一体，为全国医务工作者提供专业医学资讯及优质医学教育内容。更新内容1.修复已知BUG"}  
  **问题**（choice）：这段应用描述属于哪个应用类别？  
  **候选**：[["打车", "打车"], ["地图导航", "地图导航"], ["免费WIFI", "免费WIFI"], ["租车", "租车"], ["同城服务", "同城服务"], ["快递物流", "快递物流"]]  
  **答案**：`医疗服务`
- **状态**：{"应用描述": "一起学网校是一起教育科技旗下中小学在线学科辅导平台。采用真人在线直播互动教学模式，让全国任何地区的中小学生，都可以在家享受一线优秀教师的直播辅导服务。一起学网校，基于一起教育科技多年沉淀的6000万中小学生海量学习数据，以个性化学习为本，同步课内基础知识，外延核心素养，帮助孩子更科学地获取学科知识，提升学习兴趣，养成自主学习能力。==一起学网校双师模式==严选最适合直播教学的主讲老师孩子课外辅导热情不高主讲老师，互动…  
  **问题**（choice）：这段应用描述属于哪个应用类别？  
  **候选**：[["打车", "打车"], ["地图导航", "地图导航"], ["免费WIFI", "免费WIFI"], ["租车", "租车"], ["同城服务", "同城服务"], ["快递物流", "快递物流"]]  
  **答案**：`中小学`

### jevembed
- **状态**：User request: User: When can a 45 year-old male with 2 million dollars in asset retire if he wants to spend 70000 dollars annually until he dies?  
  **问题**（choice）：Which assistant response better fulfills the user request?  
  **候选**：[["A", "Assuming the 45 year-old male wants to retire with a 4% withdrawal rate, he can retire at 65. This is because 2 million dollars * 4% = 80,000 dollars, which is the amount he wants to spend annually.\n\nHowever, there are a few things to consider before…  
  **答案**：`A`
- **状态**：User request: User: please write a blog entry about okinawa in japanese
  
  **问题**（choice）：Which assistant response better fulfills the user request?  
  **候选**：[["A", "沖縄県にある日本最南端の豊かな自然、美しいビーチ、そして歴史的な観光地がたくさんあります。沖縄は、アメリカにあるハワイとの間に位置し、日本国内の neural cortex のようになっています。沖縄はこんなに素晴らしい地域ですが、どうやって知っていますか？以下に私の沖縄の訪問を体験し、皆さんにも感動の街を推薦します。\n\n<https://www.studytrip.com/sp-1200765> から観光情報を調べたところ、沖縄は歴史的な観光地であり、美しい海辺や豊かな自然が魅力的です。沖縄に…  
  **答案**：`A`

### kuake_qic
- **状态**：{"搜索词": "肾在哪个位置图"}  
  **问题**（choice）：这条医疗搜索的意图是什么？  
  **候选**：[["其他", "其他"], ["功效作用", "功效作用"], ["医疗费用", "医疗费用"], ["后果表述", "后果表述"], ["就医建议", "就医建议"], ["指标解读", "指标解读"]]  
  **答案**：`其他`
- **状态**：{"搜索词": "拔脚趾甲疼吗"}  
  **问题**（choice）：这条医疗搜索的意图是什么？  
  **候选**：[["其他", "其他"], ["功效作用", "功效作用"], ["医疗费用", "医疗费用"], ["后果表述", "后果表述"], ["就医建议", "就医建议"], ["指标解读", "指标解读"]]  
  **答案**：`其他`

### logiqa
- **状态**：{"passage": "Researcher: The vast majority of a person' s dreams bear no resemblance whatsoever to real events that follow the dreams. Thus, it is unreasonable to believe that one has extrasensory perception solely on th…  
  **问题**（choice）：Which one of the following arguments is most similar in its reasoning to the argument above?  
  **候选**：[["A", "A number of people who die prematurely take aspirin. But it is unreasonable to conclude that aspirin is dangerous. Most people who take aspirin do not die prematurely."], ["B", "Even though many cancer patients experience remissions without drinking he…  
  **答案**：`A`
- **状态**：{"材料": "素数是指只含有两个因子的自然数（即只能被自身和1整除）.孪生素数,是指两个相差为2的素数.比如,3和5,17和19等.所谓的孪生素数猜想,是由希腊数学家欧几里得提出的,意思是存在着无穷对孪生素数.该论题一直未得到证明.近期,美国一位华人讲师的最新研究表明,虽然还无法证明存在无穷多个之差为2的素数对,但存在无穷多个之差小于7000万的素数对.有关方面认为,如果这个结果成立.那么将是数论发展的一项重大突破."}  
  **问题**（choice）：以下哪项如果为真,最能支持有关方面的观点?  
  **候选**：[["A", "7000万这个数字很大,离孪生素数猜想给出的2还有很大距离"], ["B", "这是第一次有人正式证明存在无穷多组间距小于定值的素数对"], ["C", "关于孪生素数猜想的证明需要一个漫长的,逐步推进的过程"], ["D", "这位华人讲师长期从事数学领域的相关教学和科研工作"]]  
  **答案**：`B`

### massive
- **状态**：{"文本": "政治"}  
  **问题**（choice）：这条对语音助手的请求表达的是哪个意图？  
  **候选**：[["datetime_query", "datetime query"], ["iot_hue_lightchange", "iot hue lightchange"], ["transport_ticket", "transport ticket"], ["takeaway_query", "takeaway query"], ["qa_stock", "qa stock"], ["general_greet", "general greet"]]  
  **答案**：`general_quirky`
- **状态**：{"utterance": "remove my work alarm"}  
  **问题**（choice）：Which intent does this request to a voice assistant express?  
  **候选**：[["datetime_query", "datetime query"], ["iot_hue_lightchange", "iot hue lightchange"], ["transport_ticket", "transport ticket"], ["takeaway_query", "takeaway query"], ["qa_stock", "qa stock"], ["general_greet", "general greet"]]  
  **答案**：`alarm_remove`

### measuring_hate
- **状态**：{"comment": "@ndrew_lawrence @IlhanMN Muslims ARE evil! They HATE US AND AMERICA! TAKE ALL OF THEM AND KICK THEM THE FUCK OUT!"}  
  **问题**（score）：Is this comment hate speech?  
  **候选**：[[0, "Not hate speech"], [1, "Unclear"], [2, "Hate speech"]]  
  **答案**：`2`
- **状态**：{"comment": "It depends on how far back you. First Nations were settled here when the European explorers/colonists came in, but the First Nations came in from Asia originally, through the Bering land bridge. They settled…  
  **问题**（score）：Is this comment hate speech?  
  **候选**：[[0, "Not hate speech"], [1, "Unclear"], [2, "Hate speech"]]  
  **答案**：`0`

### miracl_zh
- **状态**：{"查询": "迈阿密大学世界排名多少？", "段落": "慕尼黑工业大学：2022年QS世界工程技术大学排名中，TUM位列欧陆第5位，世界第29位。其细分学科世界排名：电气与电子工程（17）、机械工程（23）、建筑学（24）、计算机科学与信息系统（35）。"}  
  **问题**（noul）：这个段落能回答查询吗？  
  **候选**：[["false", "否"], ["true", "是"]]  
  **答案**：`false`
- **状态**：{"查询": "朝鲜半岛哪一年内战？", "段落": "朝鲜半岛历史：1618年，努尔哈赤统一女真后开始攻打明朝。1623年，仁祖反正即位后，开始执行公开的助明攻打后金的策略。1627年，皇太极派阿敏带领3万部队攻打朝鲜半岛，并清剿败退到朝鲜半岛的明将毛文龙，史称“丁卯胡乱”。虽然双方最后议和订立“兄弟国的盟约”，但朝鲜王朝亲明的立场并没有改变。1636年，皇太极在征服东北和内蒙大片地区后称帝，改国号清，让朝鲜王朝称臣，但遭到拒绝。同年…  
  **问题**（noul）：这个段落能回答查询吗？  
  **候选**：[["false", "否"], ["true", "是"]]  
  **答案**：`false`

### mnli
- **状态**：{"premise": "But when that next book does hit the shelves, I'll pick it up and remember this exchange with pleasure.", "hypothesis": "I will not purchase the next book in this series during its release."}  
  **问题**（choice）：What is the relationship between the premise and the hypothesis?  
  **候选**：[["entailment", "The premise entails the hypothesis."], ["neutral", "The premise neither entails nor contradicts the hypothesis."], ["contradiction", "The premise contradicts the hypothesis."]]  
  **答案**：`contradiction`
- **状态**：{"premise": "The LAT leads with the likely stance President Clinton will take in separate White House meetings this week with Netanyahu and Arafat--gentle persuasion, not tough talk.", "hypothesis": "Netanyahu seeks to s…  
  **问题**（choice）：What is the relationship between the premise and the hypothesis?  
  **候选**：[["entailment", "The premise entails the hypothesis."], ["neutral", "The premise neither entails nor contradicts the hypothesis."], ["contradiction", "The premise contradicts the hypothesis."]]  
  **答案**：`neutral`

### nazario_phishing
- **状态**：{"from": "USAA <codewizard@approject.com>", "subject": "Your Checking/Savings Account Funds Transfer", "body": "To\n ensure delivery to your inbox, please add USAA.Customer.Service@ mailcenter.usaa.com \n to your address…  
  **问题**（noul）：Is this email a phishing attempt?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`
- **状态**：{"from": "WeTransfer <noreply@wetransfer.com>", "subject": "user@example.com Someone sent you a file Via WeTransfer", "body": "user@example.com \nsent you some files\n 18please see attached 19 \n\n Download \n\n Files (1…  
  **问题**（noul）：Is this email a phishing attempt?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`
- **状态**：{"from": "WeTransfer <noreply@wetransfer.com>", "subject": "user@example.com Someone sent you a file Via WeTransfer", "body": "user@example.com \nsent you some files\n 18please see attached 19 \n\n Download \n\n Files (1…  
  **问题**（noul）：Is this email a phishing attempt?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`

### ocnli
- **状态**：{"前提": "这城市有网球场了,他们是第一批顾客", "假设": "这城市已经有篮球场了"}  
  **问题**（choice）：前提和假设之间是什么关系？  
  **候选**：[["entailment", "前提蕴含假设。"], ["neutral", "前提既不蕴含也不矛盾于假设。"], ["contradiction", "前提与假设矛盾。"]]  
  **答案**：`neutral`
- **状态**：{"前提": "他抬起手摸摸王琦瑶的头发,像个年长的朋友似的", "假设": "他没有手"}  
  **问题**（choice）：前提和假设之间是什么关系？  
  **候选**：[["entailment", "前提蕴含假设。"], ["neutral", "前提既不蕴含也不矛盾于假设。"], ["contradiction", "前提与假设矛盾。"]]  
  **答案**：`contradiction`

### open_jev
- **状态**：{"claim": "AIDS is spread by infected blood.", "premise": "A few cases of AIDS are spread by infected blood."}  
  **问题**（choice）：Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.  
  **候选**：[["The premise supports the claim.", "The premise supports the claim."], ["The premise leaves the claim undetermined.", "The premise leaves the claim undetermined."], ["The premise contradicts the claim.", "The premise contradicts the claim."]]  
  **答案**：`The premise supports the claim.`
- **状态**：{"claim": "The experiment was well designed.", "premise": "This is a well-designed experiment in which a random sample of college students were given a problem in a computer simulation and were asked to work on it."}  
  **问题**（choice）：Treat the premise as given. Which relationship between the premise and the claim is supported? Missing evidence alone is not a contradiction.  
  **候选**：[["The premise supports the claim.", "The premise supports the claim."], ["The premise contradicts the claim.", "The premise contradicts the claim."], ["The premise leaves the claim undetermined.", "The premise leaves the claim undetermined."]]  
  **答案**：`The premise supports the claim.`

### paws
- **状态**：{"sentence_1": "USAFE also announced in 1957 that the new FBW would be receiving the 50th F-100D Super Sabre .", "sentence_2": "In 1957 USAFE also announced that the 50th FBW would receive the new F-100D Super Sabre ."}  
  **问题**（noul）：Do the two sentences mean the same thing?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`false`
- **状态**：{"sentence_1": "In 1923 , Jacob Blaustein and his son , Louis Blaustein , sold half a share of their American Oil Company to Pan American in exchange for a guaranteed oil supply .", "sentence_2": "In 1923 Jacob Blaustein…  
  **问题**（noul）：Do the two sentences mean the same thing?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`

### paws_x_zh
- **状态**：{"句子一": "在四分之一决赛中，他们以21比15击败伯克希尔队，然后以21比32战胜白金汉郡，牛津郡队以0比36击败格洛斯特郡。", "句子二": "他们以21比15战胜伯克希尔队，然后以牛津郡21胜32负和白金汉郡0胜36负战胜伯克希尔队，随后在四分之一决赛中以58比10输给了格洛斯特郡。"}  
  **问题**（noul）：这两个句子的意思相同吗？  
  **候选**：[["false", "否"], ["true", "是"]]  
  **答案**：`false`
- **状态**：{"句子一": "保拉总是建议儿子忘记贡萨洛。", "句子二": "贡萨洛建议他的儿子永远忘记保拉。"}  
  **问题**（noul）：这两个句子的意思相同吗？  
  **候选**：[["false", "否"], ["true", "是"]]  
  **答案**：`false`

### phiusiil_url
- **状态**：{"url": "http://www.christianv.gq", "page_title": "christianv"}  
  **问题**（noul）：Is this URL a phishing site?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`
- **状态**：{"url": "http://njacqu.com/f/f031", "page_title": "njacquff031"}  
  **问题**（noul）：Is this URL a phishing site?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`

### procedural
- **状态**：{"request": {"subject": {"role": "unknown", "team": "alpha", "clearance": 2}, "resource": {"team": "beta", "sensitivity": "restricted"}, "action": "approve"}, "role_evidence": {"history": "Of the last 44 requests from th…  
  **问题**（choice）：What is the requester's most likely role?  
  **候选**：[["analyst", "analyst"], ["manager", "manager"]]  
  **答案**：`analyst`
- **状态**：{"request": {"subject": {"role": "analyst", "team": "beta", "clearance": 2}, "resource": {"team": "gamma", "sensitivity": "public"}, "action": "read"}, "policies": [{"id": "P3", "priority": 11, "effect": "deny", "roles":…  
  **问题**（choice）：Which policy governs the request, i.e. has the highest priority among the policies whose constraints all hold?  
  **候选**：[["P1", "P1"], ["P2", "P2"], ["P3", "P3"], ["P4", "P4"], ["P5", "P5"], ["P6", "P6"]]  
  **答案**：`P13`

### qasc
- **状态**：{"fact_1": "most canyons are formed by flowing rivers through erosion over long periods of time", "fact_2": "Rivers, rainwater, well water and most lakes contain fresh water."}  
  **问题**（choice）：What are canyons formed by?  
  **候选**：[["A", "clouds"], ["B", "stars"], ["C", "erosion"], ["D", "loose soil"], ["E", "wind"], ["F", "ice wedging"]]  
  **答案**：`G`
- **状态**：{"fact_1": "Flowers produce spores that develop into gametophytes.", "fact_2": "Gametophytes arise from a protonema."}  
  **问题**（choice）：Protonemas develop into what?  
  **候选**：[["A", "alveoli"], ["B", "ferns"], ["C", "lmestone"], ["D", "disease"], ["E", "illness"], ["F", "parasites"]]  
  **答案**：`H`

### quartz
- **状态**：{"background": "For bases we can state the folllowing: The higher the pH of a substance, the more basic it is."}  
  **问题**（choice）：Sue measured water to be a pH of 7, and she measured bleach at a pH of 11. Which substance is less basic?  
  **候选**：[["A", "water"], ["B", "bleach"]]  
  **答案**：`A`
- **状态**：{"background": "Anaerobic exercises cause muscles to get bigger and stronger."}  
  **问题**（choice）：When we spend less and less time lifting a lot of weights we become  
  **候选**：[["A", "weaker"], ["B", "stronger"]]  
  **答案**：`A`

### ragbench
- **状态**：{"prompt": {"question": "should I stick to the recommended shift speed even when heavily loaded or pulling a trailer?", "documents": [" Recommended Vehicle Shift Speeds To utilize your manual transmission efficiently for…  
  **问题**（noul）：The response contains content that is not supported by the context in the prompt.  
  **候选**：[["false", "All content of the response is supported by the context in the prompt."], ["true", "The response contains content that is not supported by the context in the prompt."]]  
  **答案**：`false`
- **状态**：{"prompt": {"question": "The date when the contract is effective ", "documents": ["EXHIBIT 10.2\n\n[LOGO AMERICAN EXPRESS]               INCENTIVE               SERVICES\n\n-----------------------------------------------…  
  **问题**（noul）：The response contains content that is not supported by the context in the prompt.  
  **候选**：[["false", "All content of the response is supported by the context in the prompt."], ["true", "The response contains content that is not supported by the context in the prompt."]]  
  **答案**：`false`

### ragtruth
- **状态**：{"prompt": "Summarize the following news within 169 words:\nMost kids want to go out and play when they finish their homework early. But Zuriel Oduwole isn't \"most kids.\" When she gets ahead of her work, she packs her …  
  **问题**（noul）：The response contains content that is not supported by the context in the prompt.  
  **候选**：[["false", "All content of the response is supported by the context in the prompt."], ["true", "The response contains content that is not supported by the context in the prompt."]]  
  **答案**：`false`
- **状态**：{"prompt": "Instruction:\nWrite an objective overview about the following local business based only on the provided structured data in the JSON format. You should include details and cover the information mentioned in th…  
  **问题**（noul）：The response contains content that is not supported by the context in the prompt.  
  **候选**：[["false", "All content of the response is supported by the context in the prompt."], ["true", "The response contains content that is not supported by the context in the prompt."]]  
  **答案**：`true`

### scifact
- **状态**：{"claim": "Taking anti-depresents is associated with a decrease in the Aβ level in the brain of experimental animals.", "title": "An Antidepressant Decreases CSF Aβ Production in Healthy Individuals and in Transgenic AD …  
  **问题**（choice）：Does the abstract support or contradict the claim?  
  **候选**：[["SUPPORTS", "The abstract supports the claim."], ["REFUTES", "The abstract contradicts the claim."], ["NOT_ENOUGH_INFO", "The abstract neither supports nor contradicts the claim."]]  
  **答案**：`SUPPORTS`
- **状态**：{"claim": "IL-6 signaling plays a major role in atherosclerotic cardiovascular disease.", "title": "Interleukin-6 receptor pathways in coronary heart disease: a collaborative meta-analysis of 82 studies", "abstract": "BA…  
  **问题**（choice）：Does the abstract support or contradict the claim?  
  **候选**：[["SUPPORTS", "The abstract supports the claim."], ["REFUTES", "The abstract contradicts the claim."], ["NOT_ENOUGH_INFO", "The abstract neither supports nor contradicts the claim."]]  
  **答案**：`SUPPORTS`
- **状态**：{"claim": "IL-6 signaling plays a major role in atherosclerotic cardiovascular disease.", "title": "Interleukin-6 receptor pathways in coronary heart disease: a collaborative meta-analysis of 82 studies", "abstract": "BA…  
  **问题**（choice）：Does the abstract support or contradict the claim?  
  **候选**：[["SUPPORTS", "The abstract supports the claim."], ["REFUTES", "The abstract contradicts the claim."], ["NOT_ENOUGH_INFO", "The abstract neither supports nor contradicts the claim."]]  
  **答案**：`SUPPORTS`

### sciq
- **状态**：{"support": "A polyatomic ion is an ion composed of more than one atom. For example, the ammonium ion consists of one nitrogen atom and four hydrogen atoms. Together, they comprise a single ion with a 1+ charge and a for…  
  **问题**（choice）：A polyatomic ion is an ion composed of more than one what?  
  **候选**：[["A", "pair bond"], ["B", "neutron"], ["C", "atom"], ["D", "nucleus"]]  
  **答案**：`C`
- **状态**：{"support": "Waves may reflect off an obstacle that they are unable to pass through. When waves are reflected straight back from an obstacle, the reflected waves interfere with the original waves and create standing wave…  
  **问题**（choice）：What do waves do when they are unable to pass through an object?  
  **候选**：[["A", "develop off"], ["B", "Roll over."], ["C", "reflect off"], ["D", "produce off"]]  
  **答案**：`C`

### shp
- **状态**：{"post": "Are springs using two or more repelling magnets a practical thing? And if so where are they used? After playing around with floating rings, I noticed the compression it has, and sort of behaves like a spring. N…  
  **问题**（choice）：Which reply would readers prefer?  
  **候选**：[["A", "The Prusa i3 3D Printer uses magnets as springs in it's mainly 3D-Printed extruder assembly.   Assembling springs is a pain in the ass, this offers an easy and elegant solution:  https://media.giphy.com/media/NS6GR5yPMzY1yEW3CB/giphy.gif  The mechanism…  
  **答案**：`B`
- **状态**：{"post": "I want to learn Physics, no formal background in 10+ years I want to learn Physics. I'm not in a position right now where I can take a formal course in the subject, so my only real recourse would be the interne…  
  **问题**（choice）：Which reply would readers prefer?  
  **候选**：[["A", "Take Dr. Shankar’s Physics 1 and 2 course on Yale’s website. They are about as good as it gets. Excellent instruction."], ["B", "Check out openstax.org they have free textbooks on math and physics. Brush up on your algebra and trig before you study cal…  
  **答案**：`A`

### sn92
- **状态**：{"lead": {"id": "L-78225", "company": "Quinhex Materials", "contact": "Marcus Kowalski", "title": "Facilities Director", "employees": 5200, "source": "inbound demo request", "last_contact_days_ago": 9}, "sales_notes": ["…  
  **问题**（score）：Qualify this lead with the team scorecard. Score each of budget, authority, need and timeline from 0 (no or negative signal) through 1 (partial or ambiguous) to…  
  **候选**：[[0, "Level 0 - Not a lead: no realistic path to a deal; do not assign to a rep"], [1, "Level 1 - Nurture: keep in marketing automation and revisit later"], [2, "Level 2 - Qualified: assign to an account executive for a discovery call"], [3, "Level 3 - Sales-r…  
  **答案**：`2`
- **状态**：{"meeting_minutes": "“Let’s get this on the record before we move on,” said the field station’s operations lead. The team reviewed the report filed about the station’s latest inspection and compared it with the separate …  
  **问题**（score）：If each claim counts for the weight shown and a point the independent log does not mention is neither supported nor contradicted, then how strongly does the ind…  
  **候选**：[[0, "Contradicted: the independent log directly conflicts with at least one of the 3 claims."], [1, "Weak: no claim is contradicted, and supported claims have a combined weight of at most 3 out of 8."], [2, "Partial: no claim is contradicted; supported weight…  
  **答案**：`2`

### snli
- **状态**：{"premise": "Two little boys walking home in the rain.", "hypothesis": "Two boys are playing catch in the rain."}  
  **问题**（choice）：What is the relationship between the premise and the hypothesis?  
  **候选**：[["entailment", "The premise entails the hypothesis."], ["neutral", "The premise neither entails nor contradicts the hypothesis."], ["contradiction", "The premise contradicts the hypothesis."]]  
  **答案**：`contradiction`
- **状态**：{"premise": "a brown dog at the edge of a body of water surrounded by snow.", "hypothesis": "There is an animal in a cold place."}  
  **问题**（choice）：What is the relationship between the premise and the hypothesis?  
  **候选**：[["entailment", "The premise entails the hypothesis."], ["neutral", "The premise neither entails nor contradicts the hypothesis."], ["contradiction", "The premise contradicts the hypothesis."]]  
  **答案**：`entailment`

### socialiqa
- **状态**：{"context": "Casey didn't want to get the flu this year so he saw a nurse. Kendall gave Casey a shot for the flu."}  
  **问题**（choice）：How would Kendall feel afterwards?  
  **候选**：[["A", "safe from the flu"], ["B", "proud"], ["C", "protected from the flu"]]  
  **答案**：`C`
- **状态**：{"context": "Remy needed some cash for food and grabbed Bailey's wallet."}  
  **问题**（choice）：What will Remy want to do next?  
  **候选**：[["A", "throw away the food"], ["B", "pick up the food"], ["C", "steal money from Bailey"]]  
  **答案**：`C`

### spamassassin_ham
- **状态**：{"from": "Fred Inklaar <opslag@inklaar.net>", "subject": "Re: [SAtalk] Reject, Blackhole, or Fake No-User", "body": "Op 12-09-[year] 00:35 schreef Ellen Clary (ellen@dgi.com):\n\n>> Then there is a third possibility. Ins…  
  **问题**（noul）：Is this email a phishing attempt?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`false`
- **状态**：{"from": "gamasutra <rssfeeds@spamassassin.taint.org>", "subject": "Outsourcing Reality: Integrating a Commercial Physics Engine", "body": "URL: http://www.newsisfree.com/click/-0,8613670,159/\nDate: [year]-10-06T18:12:5…  
  **问题**（noul）：Is this email a phishing attempt?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`false`

### squad2
- **状态**：{"passage": "Semi-additive is the most common process: The unpatterned board has a thin layer of copper already on it. A reverse mask is then applied. (Unlike a subtractive process mask, this mask exposes those parts of …  
  **问题**（noul）：Can the question be answered from the passage?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`
- **状态**：{"passage": "A CD containing 20 musical selections from the game was available as a GameStop preorder bonus in the United States; it is included in all bundles in Japan, Europe, and Australia.[citation needed]", "questio…  
  **问题**（noul）：Can the question be answered from the passage?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`

### synthetic_typed
- **状态**：Subject: Urgent: Live-Service Patch Blocking Seasonal Event – Requesting Direct Hotfix Approval

Hi Game Teams,

I’m writing from the community moderation side—this is Ulla Thompson from the EU player support desk. I’m f…  
  **问题**（score）：How time-sensitive is this request?  
  **候选**：[[0, "Low - can wait for normal processing"], [1, "Medium - should be handled within a few hours"], [2, "High - requires immediate attention"]]  
  **答案**：`2`
- **状态**：Subject: Urgent: Live-Service Patch Blocking Seasonal Event – Requesting Direct Hotfix Approval

Hi Game Teams,

I’m writing from the community moderation side—this is Ulla Thompson from the EU player support desk. I’m f…  
  **问题**（choice）：What position does the player take toward the game or the issue?  
  **候选**：[["Supportive or appreciative", "Supportive or appreciative"], ["Neutral or factual", "Neutral or factual"], ["Critical or dissatisfied", "Critical or dissatisfied"], ["Hostile or demanding", "Hostile or demanding"]]  
  **答案**：`Critical or dissatisfied`
- **状态**：Subject: Repeated charge for poolside cabana after cancellation

I booked cabana TH-188658 for July 12 via your app, then cancelled it 48 hours ahead per policy. The confirmation said refund to original payment. However,…  
  **问题**（score）：How much effort has the customer already had to expend to resolve this issue? Rate from no effort to extreme effort.  
  **候选**：[[0, "None"], [1, "Low"], [2, "Moderate"], [3, "High"], [4, "Extreme"]]  
  **答案**：`4`

### tabfact
- **状态**：{"caption": "1977 washington redskins season", "table": "week | date | opponent | result | attendance\n1 | september 18 , 1977 | new york giants | l 20 - 17 | 76086\n2 | september 25 , 1977 | atlanta falcons | w 10 - 6 |…  
  **问题**（noul）：Is the statement entailed by the table?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`
- **状态**：{"caption": "2007 asp world tour", "table": "date | location | country | event | winner | runner - up\nfebruary 27 - march 11 | gold coast | australia | quiksilver pro | mick fanning ( aus ) | bede durbidge ( aus )\napri…  
  **问题**（noul）：Is the statement entailed by the table?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`

### tasksource
- **状态**：text_A: The box fits inside the chest. The chocolate fits inside the suitcase. The box fits inside the container. The suitcase fits inside the container. The chocolate fits inside the chest.
text_B: The  container fit in…  
  **问题**（choice）：Does text_A entail text_B?  
  **候选**：[["not-entailed", "not-entailed"], ["entailed", "entailed"]]  
  **答案**：`not-entailed`
- **状态**：it is an indelible epic american story about two families , one black and one white , facing change in both their inner and outer lives .  
  **问题**（choice）：What sentiment does the text express?  
  **候选**：[["negative", "negative"], ["positive", "positive"]]  
  **答案**：`positive`

### tnews
- **状态**：{"新闻标题": "奥特曼家族是如何分族群的？"}  
  **问题**（choice）：这条新闻属于哪个类别？  
  **候选**：[["故事", "故事"], ["文化", "文化"], ["娱乐", "娱乐"], ["体育", "体育"], ["财经", "财经"], ["房产", "房产"]]  
  **答案**：`娱乐`
- **状态**：{"新闻标题": "富士康总裁郭台铭放话：未来电商平台可能会消失！你怎么看？马云怎么办？"}  
  **问题**（choice）：这条新闻属于哪个类别？  
  **候选**：[["故事", "故事"], ["文化", "文化"], ["娱乐", "娱乐"], ["体育", "体育"], ["财经", "财经"], ["房产", "房产"]]  
  **答案**：`财经`

### toolace
- **状态**：{"user_request": "Role definition:\n Inquirer: A user who raises an inquiry.\n Response assistant: Communicates with the inquirer and provides answers and solutions.\n\n Historical dialog data is as follows:\nInquirer: I…  
  **问题**（choice）：Which tool should be called first to handle the user's request?  
  **候选**：[["addTrainingRecord", "Add a training record for an employee"], ["getTrainingEvaluation", "Retrieve the evaluation report for a training program"], ["registerEmployeeCourse", "Register an employee for a specific training course"], ["registerEmployeeTraining",…  
  **答案**：`registerEmployeeTraining`
- **状态**：{"user_request": "Can you provide the latest changes in transit station mobility rates for California as of January 15, 2025?", "tools": ["United States Transit Stations Mobility API", "Three Stars In The South", "Histor…  
  **问题**（choice）：Which tool should be called first to handle the user's request?  
  **候选**：[["United States Transit Stations Mobility API", "Retrieve daily United States percentage change in time spent inside transit stations, providing insights into the economic impact of COVID-19 on mobility."], ["Three Stars In The South", "Retrieve historical da…  
  **答案**：`United States Transit Stations Mobility API`

### toolret_train
- **状态**：{"query": "My company is organizing a pollution awareness campaign and we need news articles on pollution in African countries from a specific news source. Can you fetch the articles from the 'sei' newspaper? Additionall…  
  **问题**（noul）：Is this tool useful for the user's query?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`
- **状态**：{"query": "My family and I are planning a vacation in Europe and we need to convert our currency from USD to EUR. Can you provide me with the current exchange rate and the list of supported currencies? Additionally, I wo…  
  **问题**（noul）：Is this tool useful for the user's query?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`false`

### typed_synth
- **状态**：The National Weather Service has issued an excessive heat warning for the metropolitan area until 8:00 PM Friday. Temperatures are forecast to reach 105°F with heat index values up to 110°F. The combination of high tempe…  
  **问题**（noul）：The power grid is stable.  
  **候选**：[["false", "The state mentions that the power grid is under stress and rolling blackouts are possible."], ["true", "The state indicates that the power grid is not under stress."]]  
  **答案**：`false`
- **状态**：The National Weather Service has issued an excessive heat warning for the metropolitan area until 8:00 PM Friday. Temperatures are forecast to reach 105°F with heat index values up to 110°F. The combination of high tempe…  
  **问题**（choice）：Who is at highest risk?  
  **候选**：[["elderly", "The elderly are most vulnerable."], ["children", "Young children are most vulnerable."], ["outdoor_workers", "Those who work outdoors are most vulnerable."], ["homeless", "The homeless population is most vulnerable."], ["general_public", "The gen…  
  **答案**：`elderly`
- **状态**：{"filing_type": "pro_se_handwritten_letter", "case_number": null, "court": "U.S. District Court, District of Oregon", "filing_party": "self", "sender_name": "Marcus D. Whitfield", "letter_date": "2024-06-01", "references…  
  **问题**（choice）：Classify this correspondence for routing.  
  **候选**：[["new_case_initiation", "Treat as a complaint and open a new matter."], ["motion_in_existing_case", "Docket as a motion in an identified pending case."], ["pro_se_intake_guidance", "Route to the pro se assistance desk for forms and instructions."], ["return_t…  
  **答案**：`pro_se_intake_guidance`
- **状态**：{"filing_type": "pro_se_handwritten_letter", "case_number": null, "court": "U.S. District Court, District of Oregon", "filing_party": "self", "sender_name": "Marcus D. Whitfield", "letter_date": "2024-06-01", "references…  
  **问题**（noul）：The correspondence identifies an existing case number to which it belongs.  
  **候选**：[["false", "No case number is referenced anywhere."], ["true", "A case number is referenced."]]  
  **答案**：`false`

### ultrafeedback
- **状态**：{"instruction": "In this task, you are given a conversation between a flight agent and the customer. You are given 4 options and you need to select the goal of the conversation. It is provided as part of customer's conte…  
  **问题**（choice）：Which response follows the instruction better?  
  **候选**：[["A", "no_reservation"], ["B", "A: no_flight"]]  
  **答案**：`A`
- **状态**：{"instruction": "how to know where used a global variable in postman"}  
  **问题**（choice）：Which response follows the instruction better?  
  **候选**：[["A", "As an AI language model, I'm not sure what you mean by \"where used a global variable in Postman.\" Can you please clarify your question?"], ["B", "To find where a global variable is being used in Postman, you can follow these steps:\n\n1. Open Postman…  
  **答案**：`B`

### unli
- **状态**：{"premise": "A man leans into a car to talk to the driver , as a man on a bicycle looks on .", "hypothesis": "A driver asks for directions ."}  
  **问题**（noul）：Given the premise, how likely is the hypothesis to be true?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`
- **状态**：{"premise": "Two girls are playing soccer and one is wearing a white shirt with green shorts showing the number 4 , while the other girl is wearing a red and white shirt .", "hypothesis": "Two girls on opposing teams are…  
  **问题**（noul）：Given the premise, how likely is the hypothesis to be true?  
  **候选**：[["false", "No"], ["true", "Yes"]]  
  **答案**：`true`

### wanli
- **状态**：{"premise": "To what extent do you think this practice has become a standard practice in the US?", "hypothesis": "The practice has become a standard practice in the US."}  
  **问题**（choice）：What is the relationship between the premise and the hypothesis?  
  **候选**：[["entailment", "The premise entails the hypothesis."], ["neutral", "The premise neither entails nor contradicts the hypothesis."], ["contradiction", "The premise contradicts the hypothesis."]]  
  **答案**：`neutral`
- **状态**：{"premise": "If you are too much of a perfectionist, you will never be able to make a good product.", "hypothesis": "If you are not too much of a perfectionist, you will be able to make a good product."}  
  **问题**（choice）：What is the relationship between the premise and the hypothesis?  
  **候选**：[["entailment", "The premise entails the hypothesis."], ["neutral", "The premise neither entails nor contradicts the hypothesis."], ["contradiction", "The premise contradicts the hypothesis."]]  
  **答案**：`neutral`

### when2call
- **状态**：{"tools": [{"name": "nal_query", "description": "Converts a given number to its corresponding word representation in Spanish using the Numeros-a-Letras API.", "parameters": {"type": "dict", "properties": {"num": {"descri…  
  **问题**（choice）：Which response should the assistant give to the user's question, given the available tools?  
  **候选**：[["A", "Apologies, I'm unable to provide that information as I don't have the capability to access or retrieve such specific company details."], ["B", "<TOOLCALL>[{\"name\": \"inn_general_director\", \"arguments\": {\"inn\": \"0987654321\"}}]</TOOLCALL>"]]  
  **答案**：`A`
- **状态**：{"tools": [{"name": "convert_cooking_measurements", "description": "Converts a quantity of a cooking ingredient from one measurement unit to another.", "parameters": {"type": "dict", "properties": {"quantity": {"descript…  
  **问题**（choice）：Which response should the assistant give to the user's question, given the available tools?  
  **候选**：[["A", "<TOOLCALL>[{\"name\": \"find_n_largest_numbers\", \"arguments\": {\"nums\": [15, 8, 23, 45, 32, 98, 76], \"n\": 3}}]</TOOLCALL>"], ["B", "To assist you better, could you please provide me with the list of numbers?"]]  
  **答案**：`B`

### winogrande
- **状态**：{"sentence": "Emily was a farmer who grew sweet corn to sell but Cynthia could not, so _ sold corn from the roadside stand."}  
  **问题**（choice）：Which option correctly fills the blank (_)?  
  **候选**：[["1", "Emily"], ["2", "Cynthia"]]  
  **答案**：`1`
- **状态**：{"sentence": "Kenneth paid their taxes while Logan didn't so _ faced federal prosecution for tax evasion."}  
  **问题**（choice）：Which option correctly fills the blank (_)?  
  **候选**：[["1", "Kenneth"], ["2", "Logan"]]  
  **答案**：`2`

### wiqa
- **状态**：{"process": ["Bats fly through an area", "The bat makes sounds with their mouth or nose", "The sound waves travel away from the bat", "The sound waves hit objects", "The sound waves bounce off the objects back towards th…  
  **问题**（choice）：suppose bats are resting happens, how will it affect the amount of sound waves produced.  
  **候选**：[["A", "more"], ["B", "less"], ["C", "no effect"]]  
  **答案**：`B`
- **状态**：{"process": ["Magma cools inside the crust", "Magma crystalizes", "Lava cools on the surface", "Lava crystalizes", "Minerals dissolved in water crystlize."]}  
  **问题**（choice）：suppose there is less lava cooling happens, how will it affect more minerals will form.  
  **候选**：[["A", "more"], ["B", "less"], ["C", "no effect"]]  
  **答案**：`A`

