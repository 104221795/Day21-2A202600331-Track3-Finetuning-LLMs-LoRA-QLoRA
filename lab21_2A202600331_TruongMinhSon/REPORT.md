# Lab 21 - Evaluation Report

**Học viên**: TruongMinhSon - 2A202600331  
**Ngày nộp**: <7/5/2026>  
**Submission option**: B

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 cleaned samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 564)
- **GPU**: Tesla T4, 15.64 GB VRAM
- **Training cost estimate**: $0.09 (~15.5 min @ $0.35/hr)
- **Main LoRA target modules**: `['q_proj', 'v_proj']`
- **Hyperparameters**: epochs=3, lr=0.0002, cosine schedule, warmup_ratio=0.1, effective batch size=8

## 2. Rank Experiment Results

| rank   |   trainable_params |   train_time_min |   peak_vram_gb |   eval_loss |   eval_perplexity |
|:-------|-------------------:|-----------------:|---------------:|------------:|------------------:|
| base   |                  0 |             0    |         nan    |      1.835  |              6.27 |
| 8      |            1843200 |             5.4  |           3.1  |      1.5478 |              4.7  |
| 16     |            3686400 |             5.11 |           2.39 |      1.5092 |              4.52 |
| 64     |           14745600 |             5.02 |           2.52 |      1.4768 |              4.38 |

## 3. Loss Curve Analysis

Loss curve file: `results/loss_curve.png`.

The training loss curves show how quickly each LoRA rank adapts to the dataset. The final eval perplexity is used as the main quantitative comparison because mid-training eval can be expensive on T4. If `r64` has lower training loss but similar or worse eval perplexity than `r16`, that suggests diminishing returns or mild overfitting. If `r8` is close to `r16`, rank 8 may be attractive for cost-sensitive deployment.

## 4. Qualitative Comparison

### Example 1

**Prompt:** Giải thích khái niệm machine learning cho người mới bắt đầu.

**Base:** Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc học từ dữ liệu và tự cải thiện qua thời gian thay vì dựa trên các quy tắc cứng nhắc. Nhìn vào cơ bản, mục tiêu của machine learning là giúp máy học các mô hình dựa trên dữ liệu mà không cần phải được chỉ định bằng cách viết mã. Có nhiều loại machine learning khác nhau, bao gồm phân loại, dự đoán, phân tích hình ảnh, phân tích ngôn ngữ tự nhiên, ... Tùy thuộc vào mục tiêu của bạn, bạn có thể chọn loại machine learning phù hợp. Machine learning có thể được sử dụng trong nhiều lĩnh vực khác nhau, bao gồm marketing, y học, ngân hàng, ... Trong trường hợp bạn đang cố gắng hiểu một cách đơn giản nhất, có thể nghĩ về mach

**Fine-tuned r16:** Machine learning là một phương pháp tính toán mà máy tính có thể học từ dữ liệu và cải thiện chức năng của mình mà không cần phải được hướng dẫn rõ ràng. Điều này có nghĩa là máy tính có thể học được một cách tự động từ dữ liệu mà không cần phải được chỉ định các quy tắc hoặc hướng dẫn cụ thể. Ví dụ, trong trường hợp học máy, máy tính có thể học được phân loại hình ảnh bằng cách đào tạo trên một tập dữ liệu hình ảnh lớn và sau đó sử dụng các mô hình học máy để dự đoán nhãn của các hình ảnh mới. Machine learning có thể được phân loại thành ba loại chính là: phân loại, dự đoán và phân tích dữ liệu. Mỗi loại có các kỹ thuật khác nhau và có thể được sử dụng trong các ứng dụng khác nhau.

**Nhận xét:** Fill after reading outputs: improved / same / degraded and why.

### Example 2

**Prompt:** Viết đoạn code Python tính số Fibonacci thứ n.

**Base:** Để tính số Fibonacci thứ n bằng Python, bạn có thể sử dụng một hàm đệ quy như sau:

```python
def fibonacci(n):
    if n <= 0:
        return "Phải nhập số nguyên dương."
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
```

Hàm này trả về số Fibonacci thứ n bằng cách cộng số Fibonacci thứ n-1 và số Fibonacci thứ n-2. Nếu n nhỏ hơn hoặc bằng 0 hoặc bằng 2, thì hàm trả về giá trị tương ứng của số Fibonacci thứ 0 hoặc 1.

**Fine-tuned r16:** Để tính số Fibonacci thứ n trong Python, chúng ta có thể sử dụng hàm đệ quy hoặc vòng lặp. Trong bài viết này, chúng tôi sẽ sử dụng hàm đệ quy để tính toán số Fibonacci. Hàm đệ quy tính toán số Fibonacci bằng cách sử dụng phương pháp đệ quy để tính toán số Fibonacci. Hàm đệ quy này được gọi bằng cách cung cấp số Fibonacci cần tính toán, và nó sẽ trả về số Fibonacci cần tìm. Ví dụ: Để tính số Fibonacci thứ 6, bạn có thể gọi hàm như sau: `fibonacci(6)` Trong đó, `fibonacci()` là tên của hàm và 6 là số Fibonacci cần tính toán. Hàm đệ quy sẽ thực hiện các bước sau để tính toán số Fibonacci thứ n: 1. Nếu n là 0 hoặc 1, trả về n.

**Nhận xét:** Fill after reading outputs: improved / same / degraded and why.

### Example 3

**Prompt:** Liệt kê 5 nguyên tắc thiết kế UI/UX quan trọng.

**Base:** 1. **Giới thiệu dễ hiểu**: UI/UX thiết kế nên đảm bảo người dùng có thể hiểu và nhận ra được nội dung, chức năng và mục đích của trang web hoặc ứng dụng. 2. **Sự thoải mái**: UI/UX thiết kế nên tạo ra trải nghiệm người dùng thân thiện, thân thiện và dễ sử dụng. 3. **Công khai**: UI/UX thiết kế nên tạo ra một giao diện dễ dàng tiếp cận và đọc được cho tất cả người dùng, bao gồm người dùng có khả năng thăng hạng kém. 4. **Hiệu suất**: UI/UX thiết kế nên có hiệu suất cao, nhanh chóng và đáng tin cậy, đáp ứng nhu cầu của người dùng và giúp họ hoàn thành nhiệm vụ của họ một cách dễ dàng. 5. **Phân tích và cải

**Fine-tuned r16:** 1. Phân tích nhu cầu người dùng: Trước khi thiết kế UI/UX, cần xác định nhu cầu của người dùng và nhu cầu của sản phẩm. Điều này giúp thiết kế phù hợp và đáp ứng nhu cầu của người dùng. 2. Tính dễ hiểu: UI/UX phải dễ hiểu và dễ sử dụng. Người dùng nên có thể dễ dàng sử dụng sản phẩm và hiểu cách hoạt động của nó. 3. Phân cấp: UI/UX phải phân cấp rõ ràng và dễ hiểu. Các chức năng và tính năng của sản phẩm nên được sắp xếp và sắp xếp một cách hợp lý để giúp người dùng dễ dàng tìm thấy và sử dụng. 4. Tính tương thích: UI/UX phải tương thích với các thiết bị và hệ điều hành khác nhau. Người dùng nên có thể sử dụng sản phẩm trên nhiều thiết bị và hệ điều

**Nhận xét:** Fill after reading outputs: improved / same / degraded and why.

### Example 4

**Prompt:** Tóm tắt sự khác biệt giữa LoRA và QLoRA.

**Base:** LoRA (Large Language Models Optimized for Low-Rank Approximation) và QLoRA (Quantized LoRA) là hai phương pháp cải tiến được phát triển để tối ưu hóa hiệu năng và khả năng lưu trữ của mô hình LLM. Trong khi LoRA tăng cường tính hiệu quả và khả năng lưu trữ của mô hình LLM bằng cách sử dụng các tham số bổ sung có kích thước thấp để cải thiện các tham số chính, QLoRA cải thiện tốc độ và hiệu suất bằng cách sử dụng kỹ thuật phân cụm (clustering) để giảm số lượng tham số chính và làm cho mô hình dễ hơn để lưu trữ. Cụ thể, LoRA tăng cường tính hiệu quả và khả năng lưu trữ của mô hình LLM bằng cách sử dụng các tham số bổ sung có kích thước thấp để cải thiện các tham số chính,

**Fine-tuned r16:** LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải tiến để tối ưu hóa các mạng neural networks lớn hơn bằng cách sử dụng các tham số nhỏ hơn được điều chỉnh. LoRA là một phương pháp được phát triển để cải thiện việc thay đổi trọng lượng của một mạng neural lớn hơn bằng cách sử dụng các tham số nhỏ hơn được điều chỉnh để thay thế các tham số lớn của mạng neural. LoRA được sử dụng để tối ưu hóa các mạng neural lớn hơn bằng cách sử dụng các tham số nhỏ hơn được điều chỉnh để thay thế các tham số lớn của mạng neural. QLoRA là một cải tiến của LoRA được phát triển để tối ưu hóa các mạng neural lớn hơn bằng cách sử dụng các tham số nhỏ hơn được điều chỉnh và mã hóa bằng cá

**Nhận xét:** Fill after reading outputs: improved / same / degraded and why.

### Example 5

**Prompt:** Phân biệt prompt engineering, RAG, và fine-tuning.

**Base:** Prompt engineering, RAG và fine-tuning là các kỹ thuật khác nhau để cải thiện hiệu suất của mô hình AI. Prompt engineering là quá trình thiết kế câu lệnh, hoặc prompt, để tăng cường khả năng hiểu và trả lời câu hỏi của mô hình. RAG là một kỹ thuật khác để cải thiện hiệu suất của mô hình bằng cách sử dụng dữ liệu được thu thập từ nhiều nguồn khác nhau, bao gồm nguồn chính, nguồn phụ và nguồn phụ. Fine-tuning là quá trình sử dụng một mô hình đã được huấn luyện ban đầu để huấn luyện một mô hình mới cho một nhiệm vụ cụ thể bằng cách cung cấp cho mô hình một tập tin huấn luyện mới. Cả prompt engineering, RAG và fine-tuning đều là kỹ thuật hữu ích để cải thiện hiệu suất của mô hình AI và có thể đư

**Fine-tuned r16:** Prompt engineering, RAG, và fine-tuning là ba phương pháp khác nhau để tạo ra các mô hình AI được tối ưu hóa cho một bài tập cụ thể. 
- **Prompt engineering**: Đây là kỹ thuật sử dụng câu lệnh (prompt) để giúp mô hình AI hiểu và trả lời đúng câu hỏi hoặc nhiệm vụ. Điều này có thể bao gồm tạo ra các câu lệnh mô tả rõ ràng, cung cấp thêm thông tin cần thiết và loại bỏ các từ không cần thiết. 
- **RAG (Reinforcement Augmented Generation)**: RAG là một kỹ thuật để cải thiện hiệu suất của mô hình AI bằng cách sử dụng feedback của một mô hình khác (tốt hơn) để cải thiện mô hình gốc. Điều này có thể bao gồm sử dụng dữ liệu được đánh giá để nâng cao độ chính xác của mô hình hoặc sử dụng

**Nhận xét:** Fill after reading outputs: improved / same / degraded and why.


## 5. Conclusion về Rank Trade-off

The best rank by eval perplexity in this run is `r=64`. However, rank selection should balance quality and cost. Rank 8 trains fastest and uses the least adapter capacity, so it is useful for small datasets or many tenant-specific adapters. Rank 16 is usually the most practical default because it adds enough capacity for style and format adaptation while keeping VRAM and training time manageable. Rank 64 has much higher capacity, but on a 100-500 example dataset it may not improve perplexity enough to justify the extra time and memory. For production, I would choose rank 16 unless rank 64 gives a clear qualitative and perplexity improvement.

## 6. What I Learned

- LoRA rank controls adapter capacity, but higher rank does not guarantee better generalization.
- QLoRA makes this experiment possible on a Colab T4 by keeping the base model in 4-bit.
- Evaluation needs both perplexity and qualitative examples because a lower loss does not always mean a better user-facing answer.
