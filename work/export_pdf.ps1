$ErrorActionPreference = "Stop"
$source = "D:\TT paper\1024 TP Intracerebral Hemorrhage\1025Manuscript.docx"
$target = "D:\TT paper\1024 TP Intracerebral Hemorrhage\1025Manuscript.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($source, $false, $true)
    try {
        $doc.SaveAs([ref]$target, [ref]17)
    }
    finally {
        $doc.Close([ref]$false)
    }
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
