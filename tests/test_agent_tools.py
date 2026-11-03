
from unittest.mock import patch
from agent_tools import search_paper


def test_search_paper():
    with patch(
        "agent_tools.retrieve_context_with_sources"
    ) as mock_retrieve:

        mock_retrieve.return_value = (
            "模拟论文证据",
            [{"page": 6, "chunk_id": 66}]
        )

        # TODO 1：
        # 调用 search_paper()
        # query = "葡萄糖检测限是多少？"
        # all_chunks、model、embeddings
        # 暂时分别传入 []、None、[]
        result = search_paper(
            query="葡萄糖检测限是多少？",
            all_chunks=[],
            model=None,
            embeddings=[]
        )

        # TODO 2：
        # 断言结果的 context 等于 "模拟论文证据"
        assert result["context"] == "模拟论文证据"
        # TODO 3：
        # 断言结果的 sources[0]["page"] 等于 6
        assert result["sources"][0]["page"] == 6
