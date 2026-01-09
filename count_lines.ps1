$path = "c:\Users\Usuario\Desktop\Repositorio Anas\AutoOCR_FinalVersion"
$exts = @("*.py","*.html","*.css","*.js","*.yaml","Dockerfile","*.bat")
$files = Get-ChildItem -Path $path -Recurse -Include $exts | Where-Object { $_.FullName -notmatch "\\(\.git|\.gemini|venv|__pycache__|data|node_modules)\\" }
$stats = $files | Group-Object Extension | Select-Object Name, Count, @{Name="Lines"; Expression={ ($_.Group | ForEach-Object { (Get-Content $_.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines }) | Measure-Object -Sum | Select-Object -ExpandProperty Sum }}
$total = $stats | Measure-Object -Property Lines -Sum
$stats | Format-Table -AutoSize
Write-Output "TOTAL LINES: $($total.Sum)"
Write-Output "TOTAL FILES: $($files.Count)"
