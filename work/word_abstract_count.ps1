$ErrorActionPreference = "Stop"
$source = "D:\TT paper\1024 TP Intracerebral Hemorrhage\1025Manuscript.docx"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($source, $false, $true)
    $abstractStart = $doc.Content
    $abstractStart.Find.Text = "Abstract"
    $abstractStart.Find.Forward = $true
    [void]$abstractStart.Find.Execute()
    $start = $abstractStart.Start

    $keywords = $doc.Content
    $keywords.Find.Text = "Keywords:"
    $keywords.Find.Forward = $true
    [void]$keywords.Find.Execute()
    $end = $keywords.Start

    $range = $doc.Range($start, $end)
    Write-Output "abstract_words_word=$($range.ComputeStatistics(0))"
    Write-Output "abstract_characters=$($range.Characters.Count)"
    Write-Output "document_words_word=$($doc.Content.ComputeStatistics(0))"
    $doc.Close([ref]$false)
}
finally {
    $word.Quit()
}
