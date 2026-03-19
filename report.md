# 情感感知驱动的说话人语音识别系统

## 一、 选题背景与意义

### 选题背景

语音是人类最直接、最自然的交流方式。自动语音识别（ASR）技术虽已在语音内容转录方面取得了巨大成功[1]，但通常局限于对内容的解码，而忽略了其中所蕴含的丰富情感信息。这些由音高、能量、语速等构成的副语言（Paralinguistic）特征，是实现真正自然人机交互、迈向“共情计算”的关键。早期情感识别研究依赖于人工设计的声学特征（如MFCC、音高相关特征等）并结合传统分类器完成情感判别[2]，这种方法不仅耗时，且特征的有效性与泛化能力有限。随着深度学习的发展，利用神经网络从梅尔频谱图等二维声学表征中自动学习、提取分层特征成为主流范式[3]。这使得构建能够同时解码语音内容与感知说话人情感的多任务系统成为可能，也是本研究的技术切入点。

### 选题意义

本研究旨在设计并实现一个情感感知驱动的说话人语音识别系统，其理论与应用意义可从以下三个层面加以阐述。

首先，在人机交互层面，本研究关注语音交互中长期被“仅文本转录”范式弱化的副语言信息，将情感感知能力与语音内容识别过程进行协同建模与一体化输出。由于情感常通过音高、能量、语速与谱形变化等声学线索隐式表达，单纯的ASR结果难以反映说话人状态与交互语境；因此，引入情感识别模块有助于智能助手、虚拟客服与陪伴式系统形成更符合人类交流规律的反馈策略，从而推动语音交互由指令式理解走向更具语境敏感性的共情式理解。

其次，在应用拓展层面，本研究有望拓宽语音技术在真实场景中的可用边界。面向心理健康辅助、智能教育与驾驶员状态监测等任务，语音是一种低成本、非侵入式且可持续采集的行为信号，具备开展情绪与压力状态评估的天然优势。然而，当模型从训练所用的标准情感语料库迁移到实际应用场景时，口音差异、录音设备差异、说话风格及环境噪声等因素会引入显著的域偏移（Domain Shift），即训练数据分布与部署环境数据分布之间的系统性差异，导致模型在跨场景部署时性能大幅波动。基于此，本研究通过构建端到端的处理与评估流程，并在统一协议下开展域内与跨域对比实验，旨在为情感感知语音系统的稳健性评估与工程化落地提供可验证的技术依据。

最后，在理论探索层面，本项目的核心贡献在于对两种主流深度学习范式进行系统对比与机制性分析：其一是从零开始训练、针对情感任务高度特化的定制模型（即本项目中的CNN-BiLSTM-Attention架构）；其二是基于大型预训练模型（如Whisper）的迁移学习方案。初步实验往往显示，迁移学习路线能够在有限标注数据与复杂分布条件下取得更稳定的识别效果；与此同时，自建模型即便在当前阶段准确率相对较低，仍可作为受控基线用于刻画特征表示、模型容量与正则化策略对泛化性能的影响边界，并为后续改进提供可解释的误差分析参照。通过量化评估两者在域内准确率与跨域泛化能力上的表现差异，并综合比较训练成本与部署约束，本研究旨在为语音情感识别领域长期存在的“跨数据集泛化性能差”问题提供实证依据与方法论参考。

## 二、 国内外研究现状

### 国外研究现状

语音情感识别（Speech Emotion Recognition, SER）的研究经历了从传统模式识别到端到端深度学习的范式演进。

#### 1. 早期阶段：基于人工特征与传统分类器（2000-2014）

早期研究主要依赖专家知识与显式特征工程，通常借助openSMILE、Praat等工具提取韵律特征（音高F0、能量、时长）、音质特征（共振峰、谐波噪声比HNR）以及频谱相关特征（MFCC、LPC、频谱质心），并进一步结合SVM、随机森林、GMM-HMM等传统分类器完成情感判别。尽管该路线在小规模、受控语料上具有一定有效性，但其性能高度依赖特征设计与先验经验，难以系统刻画情感表达中的非线性模式与长程时序依赖关系，同时在跨说话人、跨录音环境的场景下往往表现出显著的泛化不足。

#### 2. 深度学习时代：CNN+LSTM混合架构（2015-2020）

随着深度学习的发展，研究者逐步转向从梅尔频谱图等时频表征中端到端学习情感判别特征，并形成了以“卷积特征提取—时序建模—注意力聚合”为代表的经典范式。以Fayek等人（2017）对多种深度网络架构的系统评估为例，模型通常先由CNN捕获频谱图的局部模式（如共振峰形态与音高突变）[3]，再由（双向）LSTM对时间维上的上下文依赖进行建模[4]，最后通过注意力机制对关键时间帧进行自适应加权，从而在固定维度表示与信息保留之间取得更好的平衡。该类混合架构的优势在于同时兼顾局部声学线索与全局动态变化，因而在多种SER基准上相较于纯CNN或纯RNN模型表现出更稳定的增益。

#### 3. 注意力机制与架构创新（2018-2023）

为提升模型性能，研究者引入了多种注意力机制和架构创新。其一，残差连接与通道注意力逐渐成为深层网络训练与特征选择的重要手段。He等人（2016）提出的残差网络通过引入恒等映射缓解深度网络的梯度消失问题，使得更深的特征提取器能够稳定优化[5]；与此同时，Hu等人（2018）提出的Squeeze-and-Excitation机制通过全局信息压缩与通道重标定，动态突出更具判别力的声学通道，从而在SER任务中提升了特征表征的有效性[6]。其二，多头注意力机制被用于增强模型对多维情感线索的并行建模能力。与单头注意力仅能学习单一关注模式不同，多头结构允许模型在多个子空间中同时捕获音高、能量与节奏等不同维度的情感信号，因而在表达能力与可解释性方面均具有优势[7]。其三，面向长期依赖建模的结构创新亦不断出现，例如Islam等人（2023）提出将图卷积网络与HuBERT表征进行集成，通过图结构显式刻画语音帧之间的非局部关系，从而补足传统CNN与LSTM在长程依赖捕获方面的不足[8]。

#### 4. 前沿方向：大型预训练模型与迁移学习（2020至今）

近年来，语音处理领域的研究焦点已转向基于Transformer的大型预训练模型。这类模型通过在海量无标注或弱标注数据上进行自监督或弱监督预训练，学习到对内容、说话人、环境等变化因素具有高度鲁棒性的通用声学表征。其中，Wav2Vec 2.0通过对比学习在无标注数据上预训练，学习离散化的声学单元表示，在多个SER数据集上达到了当时的最优性能[9]；HuBERT则采用掩码预测任务，通过聚类生成伪标签并迭代优化，在IEMOCAP等基准上表现优于Wav2Vec 2.0[10]。OpenAI的Whisper模型则采用了不同的预训练策略，其在68万小时多语言、多领域音频上进行多任务ASR训练，编码器因而学习到对语言、口音、录音环境等因素高度不变的通用表征[1]。Li等人（2023）在INTERSPEECH上发表的研究表明，Whisper在情感语音上表现出较强的鲁棒性，且情感与ASR任务之间存在相互影响关系[11]。

在最新的研究趋势中（2024-2026），情感模糊性建模成为重要方向。Zhang等人（2026）提出的VoxEmo基准涵盖35个语料库、15种语言，并引入分布感知软标签协议以解决情感标注的主观性与模糊性问题[12]；Yu等人（2026）进一步将情感识别重构为分布式推理任务，提出模糊性感知目标函数使预测与人类感知分布对齐[13]。与此同时，轻量化与效率优化亦受到广泛关注。Ahmed等人（2026）提出的SpectroFusion-ViT仅包含2.04M参数却在SUBESCO数据集上达到92.56%准确率[14]；Zhang等人（2026）从生理机制出发构建语音谱时表征，为提升鲁棒性与可解释性提供了新的建模路径[15]；Su等人（2026）提出的PTS-SNN脉冲神经网络在IEMOCAP上达到73.34%准确率的同时，推理能耗仅为0.35mJ，为边缘设备上的实时情感识别提供了可能[16]。此外，Dong等人（2026）首次系统评估了11种测试时自适应（TTA）方法在SER中的表现，发现无反向传播方法最具前景，而传统的熵最小化与伪标签策略因情感模糊性而失效，这为解决跨域泛化问题提供了新的思路[17]。与此同时，面向开放环境鲁棒性的研究也开始关注对抗扰动与安全性评估，Facchinetti等人（2024）对SER模型的对抗攻击进行了系统实验，为风险分析与防御设计提供了参考框架[18]。从更早期的工程探索来看，Hifny与Ali（2019）已在资源受限场景下讨论高效情感识别网络设计，为后续轻量化路线提供了经验依据[19]。

### 国内研究现状

在语音情感识别方向，国内研究紧跟国际前沿，并呈现出以下特点。其一，围绕中文韵律与声调特性，研究者持续推进中文情感语音数据库的构建与开放共享，其中CASIA等语料以多说话人、类别覆盖较全等特征，成为中文SER研究的重要基准[24]；与此同时，ESD等更大规模的中英双语情感数据集为跨语言迁移与泛化评估提供了新的数据支撑[20] [21]。其二，在模型方法层面，国内工作一方面沿用并改进CNN+LSTM、Transformer等主流结构，另一方面更强调与中文语言学特性相结合的任务设定，例如探索声调与情感的联合建模、多任务学习与跨语言迁移学习，从而在标注数据有限的现实约束下提升模型可用性与稳健性。其三，在产业落地层面，语音情感识别已在智能客服质检、在线教育与心理健康监测等场景中出现规模化应用需求，但真实环境中的口音多样性、语码转换以及自然对话场景与实验室演绎数据之间的分布差异，仍持续制约模型在开放场景下的泛化性能，这也进一步凸显了跨域鲁棒性研究与工程化评估体系构建的重要性。

## 三、 毕业论文的进度安排

本研究计划以一学期（约16周）为周期推进，整体采用“先理论调研、再数据准备与模型实现、继而开展训练评估、最终完成系统集成与论文写作”的路线，以确保技术方案论证、实验验证与工程落地之间形成闭环。

在时间安排上，研究前两周将集中完成文献调研与方案细化，通过精读SER与预训练语音模型相关文献并对标课程/任务书要求，完成开题报告的撰写与定稿。
随后在第三至第五周进入数据准备阶段，完成ESD[20] [21]、RAVDESS[22]、TESS[23]、CASIA[24]等公开情感语音数据的整理与清洗，并统一标签空间与音频格式，形成可复现的训练/验证/测试划分，同时实现包含降噪、静音切除、重采样与定长裁剪在内的标准化预处理流程。
第六至第九周将完成两条技术路线的模型实现与联调：一方面实现基于CNN+BiLSTM+Attention的自建情感识别模型，另一方面实现基于Whisper预训练编码器的迁移学习模型（WhisperEmotionHead），并打通特征提取、训练脚本与检查点管理。
第十至第十一周将开展模型训练与超参数调优，系统记录损失、准确率与F1等指标曲线，并据验证集表现对学习率、正则化强度与损失函数等关键因素进行迭代调整。
第十二至第十三周将推进系统集成与界面实现，完成一个支持实时录音与音频上传的Web交互界面，将训练好的模型封装为推理服务并实现转录与情感结果的可视化展示。
第十四至第十五周集中完成论文写作与结果分析的定稿工作，围绕实验设置、对比结论与局限性讨论形成完整论证链条；
最后一周将根据指导意见进行修改润色，并完成答辩材料（PPT与演示流程）的准备，以确保论文质量与展示效果。

## 四、 毕业论文的研究方法

本研究将采用理论研究与实验研究相结合的方法，以确保模型设计的理论完备性与实验结论的可验证性。在理论层面，研究将系统梳理深度学习、数字信号处理、自动语音识别（ASR）与语音情感识别（SER）的关键理论，并围绕卷积神经网络（CNN）、循环神经网络（LSTM）、注意力机制与Transformer架构等方法的基本原理、适用边界与工程实现要点展开学习与分析，从而为后续模型设计提供统一的理论框架。

在实验层面，研究将以公开情感语音数据集为基础开展数据驱动的验证。具体而言，实验将选取RAVDESS、CASIA、TESS与ESD等数据集构建训练与评估语料，并通过音频整理、降噪、静音切除、重采样与定长处理实现跨数据源的输入规范化；随后分别采用两条技术路线构建情感识别模型，即从零训练的CNN+BiLSTM+多头注意力自建模型，以及基于Whisper预训练编码器的迁移学习模型。前者旨在通过残差连接与SE通道注意力强化局部声学模式的提取能力，并利用BiLSTM与多头注意力完成时序建模与关键帧聚合；后者则通过冻结Whisper Encoder并训练轻量级分类头，利用大规模预训练获得的通用声学表征提升跨域鲁棒性。为了提高实验效率与可复现性，研究将采用特征缓存策略，将Whisper编码器输出的pooled特征一次性提取并落盘，从而在训练阶段避免重复的编码器前向计算。

最后，在量化分析层面，研究将综合采用准确率、加权F1分数与混淆矩阵等指标对模型性能进行评估，并结合置信度分布对模型输出的确定性进行分析。进一步地，实验将通过域内测试与跨域测试对比检验模型的泛化能力差异，从而在经验层面回答“从零训练模型与迁移学习模型在情感识别任务中的性能—泛化权衡”这一核心研究问题。

## 五、 毕业论文的主要内容

本毕业论文将围绕“情感感知驱动的说话人语音识别系统”的设计、实现与评估展开，正文总体按照“问题提出—技术综述—系统与数据—模型方法—实验验证—系统演示—总结展望”的逻辑组织，力求在理论阐释与工程实现之间建立清晰映射，并以可复现实验结果支撑结论。

在内容组织上，第一章作为绪论，将从语音交互的自然性需求出发，阐明将情感感知引入语音识别与交互系统的研究动机与应用价值，并在此基础上梳理国内外研究进展，明确本研究的核心问题与总体贡献。第二章将对支撑本研究的关键技术进行系统综述，涵盖语音信号处理与特征表示方法、深度学习常用网络结构与正则化策略，以及Transformer与预训练语音模型的基本原理，从而为后续模型设计与实验设置提供统一的技术背景。

第三章将给出系统总体设计与数据预处理方案，具体说明训练与推理两阶段的流程划分、数据来源与标签映射策略，并围绕降噪、静音切除、重采样、归一化与定长处理等步骤构建一致的输入规范；同时，本章将阐述特征提取的实现细节，包括自建模型所需的Mel频谱与MFCC特征，以及基于Whisper的log-mel特征与特征缓存机制，以确保后续训练迭代的效率与可复现性。第四章作为论文核心，将分别描述两条情感识别技术路线的模型结构与实现要点：其一是从零训练的CNN+BiLSTM+Attention架构，通过残差与通道注意力强化局部特征提取、通过双向时序建模与多头注意力完成关键帧聚合；其二是基于Whisper预训练编码器的迁移学习方案，通过冻结编码器并训练轻量级分类头，以利用通用声学表征提升跨域鲁棒性，并对两者在参数规模、训练成本与适用场景上的差异进行机制性分析。

第五章将围绕实验设计与结果分析展开，详细说明训练环境与超参数设置，展示训练收敛过程，并以准确率、加权F1与混淆矩阵等指标对两种模型在域内与跨域场景下的性能进行量化评估；同时，结合置信度分布与典型错误案例对模型行为进行解释性分析，并通过必要的消融实验验证关键模块（如残差连接、SE通道注意力与多头注意力）的贡献。第六章将介绍系统实现与演示，包括推理流水线的封装方式、模型动态切换机制以及基于Gradio的交互界面设计，并展示转录文本与情感结果的可视化呈现，从而验证系统在真实交互流程中的可用性。第七章将对全文工作进行总结，归纳主要结论与创新点，讨论当前方案在数据规模、情感粒度与跨域泛化方面的局限，并在最新研究趋势的背景下提出后续可行的改进方向。

### 本研究尚未解决的问题

尽管本研究在情感感知驱动的语音识别系统设计与实现方面取得了阶段性进展，但受限于时间、数据与计算资源等客观条件，仍存在以下亟待深入探索的问题。

首先，在跨域泛化能力方面，当前模型在跨数据集、跨语言与跨录音环境场景下的性能仍存在显著波动。尽管基于Whisper的迁移学习方案在一定程度上缓解了域偏移问题，但面对真实场景中的口音多样性、自发对话与演绎语音的韵律差异、以及低信噪比环境下的声学退化，模型的鲁棒性仍有待进一步提升。特别是在中文情感识别任务中，声调与情感的交互作用尚未得到充分建模，这可能导致模型在处理声调语言时出现系统性偏差。未来研究可考虑引入域自适应技术（如对抗训练、测试时自适应）或构建更大规模的多域联合训练语料，以增强模型的跨场景迁移能力。

其次，在情感表达的细粒度建模层面，本研究采用的六类离散情感标签（快乐、愤怒、悲伤、中性、恐惧、惊讶）难以全面刻画人类情感的连续性与模糊性特征。现有标注协议通常基于多数投票或单一标注者判断，忽略了情感感知的主观性与标注者间的分歧，这可能导致训练数据中存在标签噪声与边界模糊样本。近期研究已开始探索分布感知的软标签建模与情感模糊性量化方法[12][13]，但这些方法在本研究中尚未实现。此外，当前系统仅关注语音模态，而真实交互场景中的情感表达往往是多模态的（如面部表情、肢体语言与语音的协同），单模态建模的局限性可能制约系统在复杂场景下的判别能力。

再次，在模型可解释性与误差分析方面，尽管本研究通过混淆矩阵与置信度分布对模型行为进行了初步分析，但对于模型决策的内在机制、关键声学线索的定位以及错误样本的系统性归因仍缺乏深入探讨。特别是对于深度神经网络而言，其"黑箱"特性使得难以直观理解模型在何种声学模式下做出特定情感判断，这在一定程度上限制了模型的可信度与在高风险场景（如心理健康评估）中的应用潜力。未来工作可引入注意力可视化、显著性分析或可解释性增强架构（如原型网络），以提升模型决策的透明度与可审计性。

最后，在系统工程化与实时性优化方面，当前系统虽已实现基于Gradio的交互界面，但在推理延迟、内存占用与边缘设备部署等维度仍存在优化空间。特别是Whisper编码器的参数规模较大（约240M），在资源受限的移动端或嵌入式设备上难以实现低延迟推理。尽管本研究通过特征缓存策略提升了训练效率，但在实时流式推理场景下，如何在保持识别精度的前提下实现模型压缩、量化与加速，仍是工程落地的关键挑战。此外，系统的安全性与鲁棒性评估（如对抗样本攻击、隐私保护）亦未纳入当前研究范围，这些问题在面向真实应用部署时需要系统性考量。

## 六、 毕业论文的主要参考文献

[1] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In Proceedings of the 40th International Conference on Machine Learning (ICML'23), Vol. 202. JMLR.org, Article 1182, 28492–28518.
[2] Akçay, M. B., & Oğuz, K. (2020). Speech emotion recognition: Emotional models, databases, features, preprocessing methods, supporting modalities, and classifiers. Speech Communication, 116, 56-76.
[3] Fayek, H. M., Lech, M., & Cavedon, L. (2017). Evaluating deep learning architectures for speech emotion recognition. Neural Networks, 92, 60-68.
[4] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
[5] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 770–778).
[6] J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 2018, pp. 7132-7141, doi: 10.1109/CVPR.2018.00745.
[7] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17). Curran Associates Inc., Red Hook, NY, USA, 6000–6010.
[8] Islam, S., Haque, M. M., & Sadat, A. J. M. (2023). Capturing spectral and long-term contextual information for speech emotion recognition using deep learning techniques. arXiv preprint arXiv:2308.04517.
[9] Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. Wav2vec 2.0: a framework for self-supervised learning of speech representations. In Proceedings of the 34th International Conference on Neural Information Processing Systems (NIPS '20). Curran Associates Inc., Red Hook, NY, USA, Article 1044, 12449–12460.
[10] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. IEEE/ACM Trans. Audio, Speech and Lang. Proc. 29 (2021), 3451–3460. https://doi.org/10.1109/TASLP.2021.3122291
[11] Li, Y., Zhao, Z., Klejch, O., Bell, P., & Lai, C. (2023). ASR and emotional speech: A word-level investigation of the mutual impact of speech and emotion recognition. arXiv preprint arXiv:2305.16065.
[12] Zhang, H., Chou, H., Narayanan, S.S., & Hain, T. (2026). VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs.
[13] Yu, X., Dong, J., Honorio, J., Ghosh, A., Jia, H., & Dang, T. (2026). Disentangling Reasoning in Large Audio-Language Models for Ambiguous Emotion Prediction. arXiv preprint arXiv:2603.08230.
[14] Ahmed, F., Chowdhury, R. H., Moon, F. T. Z., & Ahmed, S. (2026). SpectroFusion-ViT: A Lightweight Transformer for Speech Emotion Recognition Using Harmonic Mel-Chroma Fusion. arXiv preprint arXiv:2603.00746.
[15] Zhang, X., Cao, L., Yang, R., & Wu, Z. (2026). Learning Physiology-Informed Vocal Spectrotemporal Representations for Speech Emotion Recognition. arXiv preprint arXiv:2602.13259.
[16] Su, X., Wang, H., & Zhang, Q. (2026). PTS-SNN: A Prompt-Tuned Temporal Shift Spiking Neural Networks for Efficient Speech Emotion Recognition. arXiv preprint arXiv:2602.08240.
[17] Dong, J., Jia, H., & Dang, T. (2026). Test-Time Adaptation for Speech Emotion Recognition. arXiv preprint arXiv:2601.16240.
[18] Facchinetti, N., Simonetta, F., & Ntalampiras, S. (2024). A systematic evaluation of adversarial attacks against speech emotion recognition models. Intelligent Computing, 3, 0088.
[19] Hifny, Y., & Ali, A. (2019, May). Efficient arabic emotion recognition using deep neural networks. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6710-6714). IEEE.
[20] Zhou, K., Sisman, B., Liu, R., & Li, H. (2021, June). Seen and unseen emotional style transfer for voice conversion with a new emotional speech dataset. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 920-924). IEEE.
[21] Kun Zhou, Berrak Sisman, Rui Liu, and Haizhou Li. 2022. Emotional voice conversion: Theory, databases and ESD. Speech Commun. 137, C (Feb 2022), 1–18. https://doi.org/10.1016/j.specom.2021.11.006
[22] Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PloS one, 13(5), e0196391.
[23] Pichora-Fuller, M. K., & Dupuis, K. (2020). Toronto emotional speech set (TESS) (Version 1.0) [Data set]. Borealis. https://doi.org/10.5683/SP2/E8H2MF
[24] Westwest. (2024). CASIA Dataset (Speech Emotion Recognition Version) [Data set]. ModelScope. https://www.modelscope.cn/datasets/Westwest/CASIA
