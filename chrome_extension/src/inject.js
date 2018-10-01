chrome.extension.sendMessage({}, function(response) {
	var readyStateCheckInterval = setInterval(function() {
		if (document.readyState === "complete") {
			clearInterval(readyStateCheckInterval);

			var setTimeOut;
			$element = $('#bodyContent a');

			$element.on('mouseenter', function(event){
				var obj = $(this);

				setTimeOut = setTimeout(function(){
					if(obj.children('img').length === 0)
					{
						performWork(obj, event);
					}
				}, 200);

			}).on('mouseleave', function(){
				clearTimeout(setTimeOut);
				clearData();
			});

			$('body').on('click', function(){
				clearData();
			});

			function clearData()
			{
				$('#popup-notification').remove();
				$(this).attr('title', $(this).attr('data-title'));
			}

			function waitUntil(isready, success, error, count, interval){
			    if (count === undefined) {
			        count = 1000;
			    }
			    if (interval === undefined) {
			        interval = 20;
			    }
			    if (isready()) {
			        success();
			        return;
			    }
			    // The call back isn't ready. We need to wait for it
			    setTimeout(function(){
			        if (!count) {
			            // We have run out of retries
			            if (error !== undefined) {
			                error();
			            }
			        } else {
			            // Try again
			            waitUntil(isready, success, error, count-1, interval);
			        }
			    }, interval);
			}

			function sleep(ms) {
				return new Promise(resolve => setTimeout(resolve, ms));
			}

			function performWork(element, event)
			{
				//Clear the data
				clearData();
				//Remove the title attribute    
				element.removeAttr('title');

				var baseURL = window.location.origin, //Get the URL
				    url     = element.attr('href'), //Get the href of the link
				    fullURL = baseURL+url, //Combine those two together to get the link
				    origin_title = $(document).find('title').text().replace(' - Wikipedia', ''),//String(window.location.href).replace(/.*\/wiki\//, ''),
				    //target_title = String(url).replace(/.*\/wiki\//, ''),
				    origin_content = "",
				    origin_context_a = "",
				    origin_context_p = "";

				// console.log(element);
				// console.log(element.parent());

				$('p').each(function() {
					if ($(this).is(element.parent())) {
						origin_context_a = element[0].outerHTML.replace(/\n/g, ' ');
						origin_context_p = $(this).html().replace(/\n/g, ' ');
					}
				    origin_content += $(this).html().replace(/\n/g, ' ');//.replace(/\[0-9+\]/g, '');
				});

				$.get(fullURL, function(data) {
				    // var target_content = $(data).find('#bodyContent').find('p').html();
				    var target_title = $(data).filter('title').text().replace(' - Wikipedia', ''),
				        target_content = "";

				    $(data).find('p').each(function() {
					    target_content += $(this).html().replace(/\n/g, ' ');//.replace(/\[0-9+\]/g, '');
					});
				        // cssRules = {
				        // 	position: 'fixed', 
				        // 	top: 10+"px", 
				        // 	left: 10+"px", 
				        // 	right: 10+"px",
				        // 	maxHeight: 500+"px",
				        // 	padding: 1.4+"em",
				        // 	background: "#fff",
				        // 	boxShadow: "2px 2px 15px rgba(50,50,50,.75)",
				        // 	zIndex: 100
				        // };

				    // $('<div id="popup-notification">'+content+'abcdefg'+'</div>').css(cssRules).appendTo('body').fadeIn();

					if (String(url).includes('/wiki/') && !target_title.includes('#')) {

						var send_dict = {origin_title: origin_title,
										 target_title: target_title,
										 origin_content: origin_content,
										 target_content: target_content,
										 origin_context_a: origin_context_a,
										 origin_context_p: origin_context_p};

						var hover_text = $.ajax
							({
								type: "POST",
								// TODO: remove following after done with local testing
								url: "http://localhost:5000/api/m1/1234",
								// url: "https://wikontext.us/api/m1/1234",
								dataType: "text",
								async: false,
								data: JSON.stringify(send_dict),
								contentType: "application/json",
								success: function (result) { }
							}).responseText;

						waitUntil(function(){
							    return $('a.mwe-popups-extract p').length > 0;
							}, function(){
							    // $('a.mwe-popups-extract p').append(hover_text);

							    // $('a.mwe-popups-extract p').css("height", "500px");
							    // $('a.mwe-popups-extract p').css("float", "left");
							    // $('a.mwe-popups-extract p').css("overflow-y", "scroll");

				                $('a.mwe-popups-extract p').html(hover_text);
							}, function(){
							    console.log("Page preview taking too long to load.");
						});
					}
					// $('Test').appendTo($('a.mwe-popups-extract p'));
				});
			}
		}
	}, 10);
});